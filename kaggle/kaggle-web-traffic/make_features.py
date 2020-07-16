import pandas as pd
import numpy as np
import os.path
import os
import argparse
import time

import extractor
from feeder import VarFeeder
import numba
from typing import Tuple, Dict, Collection, List
import warnings

warnings.filterwarnings('ignore')

def read_cached(name) -> pd.DataFrame:
    """
    Reads csv file (maybe zipped) from data directory and caches it's content as a pickled DataFrame
    :param name: file name without extension
    :return: file content
    """
    cached = 'data/%s.pkl' % name
    sources = ['data/%s.csv' % name, 'data/%s.csv.zip' % name]
    if os.path.exists(cached):
        return pd.read_pickle(cached)
    else:
        for src in sources:
            if os.path.exists(src):
                df = pd.read_csv(src)
                df.to_pickle(cached)
                return df


def read_all() -> pd.DataFrame:
    """
    Reads source data for training/prediction
    """
    def read_file(file):
        df = read_cached(file).set_index('Page')
        # df.to_csv("./data/"+file+'.csv')
        # df.columns = df.columns.astype('M8[D]')
        # print(df.head())
        return df

    # Path to cached data
    path = os.path.join('data', 'all.pkl')
    if os.path.exists(path):
        df = pd.read_pickle(path)
        # print("read_all df",path,df.head(),df.shape)
    else:
        # Official data
        df = read_file('train_2')
        # Scraped data
        scraped = read_file('2017-08-15_2017-09-11')
        # print('scraped',scraped.head(),scraped.shape)
        # Update last two days by scraped data
        # todo 一个是145063行，一个是144301行为什么可以直接拼接呢。应该是只是增加了2天的数据
        # print('df.shape,scraped.shape',df.shape,scraped.shape)
        df[pd.Timestamp('2017-09-10')] = scraped['2017-09-10']
        df[pd.Timestamp('2017-09-11')] = scraped['2017-09-11']

        df = df.sort_index()
        # print('df.shape,scraped.shape',df.shape,scraped.shape,df.head(),df.tail())
        # Cache result
        # df.to_csv('./data/all.csv')
        df.to_pickle(path)
    return df

# todo:remove
def make_holidays(tagged, start, end) -> pd.DataFrame:
    def read_df(lang):
        result = pd.read_pickle('data/holidays/%s.pkl' % lang)
        return result[~result.dw].resample('D').size().rename(lang)

    holidays = pd.DataFrame([read_df(lang) for lang in ['de', 'en', 'es', 'fr', 'ja', 'ru', 'zh']])
    holidays = holidays.loc[:, start:end].fillna(0)
    result =tagged[['country']].join(holidays, on='country').drop('country', axis=1).fillna(0).astype(np.int8)
    result.columns = pd.DatetimeIndex(result.columns.values)
    return result


def read_x(start, end) -> pd.DataFrame:
    """
    Gets source data from start to end date. Any date can be None
    """
    df = read_all()
    # User GoogleAnalitycsRoman has really bad data with huge traffic spikes in all incarnations.
    # Wikipedia banned him, we'll ban it too
    # todo 去掉脏数据
    bad_roman = df.index.str.startswith("User:GoogleAnalitycsRoman")
    # print("bad_roman",bad_roman,bad_roman.shape,sum(bad_roman))
    df = df[~bad_roman]
    # print('df.shape',df.shape)
    if start and end:
        return df.loc[:, start:end]
    elif end:
        return df.loc[:, :end]
    else:
        return df


@numba.jit(nopython=True)
def single_autocorr(series, lag):
    """
    # todo 一组相差lag天的数据的相关系数
    Autocorrelation for single data series
    :param series: traffic series
    :param lag: lag, days
    :return:
    """
    s1 = series[lag:]
    s2 = series[:-lag]
    # print('s1,s2',s1.shape,s2.shape)
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0


@numba.jit(nopython=True)
def batch_autocorr(data, lag, starts, ends, threshold, backoffset=0):
    """
    # todo 计算每个页面访问量与滑窗lag天的相关系数；一个页面一个系数
    Calculate autocorrelation for batch (many time series at once)
    :param data: Time series, shape [n_pages, n_days]
    :param lag: Autocorrelation lag
    :param starts: Start index for each series
    :param ends: End index for each series
    :param threshold: Minimum support (ratio of time series length to lag) to calculate meaningful autocorrelation.
    :param backoffset: Offset from the series end, days.
    :return: autocorrelation, shape [n_series]. If series is too short (support less than threshold),
    autocorrelation value is NaN
    """
    n_series = data.shape[0]
    n_days = data.shape[1]
    max_end = n_days - backoffset
    corr = np.empty(n_series, dtype=np.float64)
    support = np.empty(n_series, dtype=np.float64)
    for i in range(n_series):
        # todo 每个页面的访问量。每个页面访问量>0的天数
        series = data[i]
        end = min(ends[i], max_end)
        real_len = end - starts[i]
        # print("real_len",real_len,end,max_end,starts[i])
        support[i] = real_len/lag
        if support[i] > threshold:
            series = series[starts[i]:end]
            c_365 = single_autocorr(series, lag)
            c_364 = single_autocorr(series, lag-1)
            c_366 = single_autocorr(series, lag+1)
            # Average value between exact lag and two nearest neighborhs for smoothness
            corr[i] = 0.5 * c_365 + 0.25 * c_364 + 0.25 * c_366
        else:
            corr[i] = np.NaN
    return corr #, support


@numba.jit(nopython=True)
def find_start_end(data: np.ndarray):
    """
    Calculates start and end of real traffic data. Start is an index of first non-zero, non-NaN value,
     end is index of last non-zero, non-NaN value
    :param data: Time series, shape [n_pages, n_days]
    :return:
    """
    n_pages = data.shape[0]
    n_days = data.shape[1]
    start_idx = np.full(n_pages, -1, dtype=np.int32)
    end_idx = np.full(n_pages, -1, dtype=np.int32)
    for page in range(n_pages):
        # scan from start to the end
        for day in range(n_days):
            if not np.isnan(data[page, day]) and data[page, day] > 0:
                start_idx[page] = day
                break
        # reverse scan, from end to start
        for day in range(n_days - 1, -1, -1):
            if not np.isnan(data[page, day]) and data[page, day] > 0:
                end_idx[page] = day
                break
    # todo start_idx= [0 2 0 ... 0 0 0], end_idx= [804 804 804 ... 804 804 804]
    #  返回的是每个网页浏览量非0的 起止日期
    # print(f"start_idx=start_idxz{start_idx}, end_idx={end_idx}")
    return start_idx, end_idx


def prepare_data(start, end, valid_threshold) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    todo 读取数据，找到每个页面的访问量>0起止时间，如果一个网页访问量>0的天数太少直接删掉
    Reads source data, calculates start and end of each series, drops bad series, calculates log1p(series)
    :param start: start date of effective time interval, can be None to start from beginning
    :param end: end date of effective time interval, can be None to return all data
    :param valid_threshold: minimal ratio of series real length to entire (end-start) interval. Series dropped if
    ratio is less than threshold
    :return: tuple(log1p(series), nans, series start, series end)
    """
    df = read_x(start, end)
    # print("prepare_data, df",df.head())
    starts, ends = find_start_end(df.values)
    # boolean mask for bad (too short) series
    # todo 如果一个网页的有效的数据太少就删除这个网页
    page_mask = (ends - starts) / df.shape[1] < valid_threshold
    print("Masked %d pages from %d" % (page_mask.sum(), len(df)))
    inv_mask = ~page_mask
    df = df[inv_mask]
    nans = pd.isnull(df)
    # print("nans",nans)
    # print("starts,ends.shape",starts.shape,ends.shape,starts)
    # todo log1p(x)=log(1 + x)
    #  返回已经去掉访问天数太少的网页暂时称为有效网页并做log2变换，这些数据有哪些为空，有效网页访问量非0的起止时间
    return np.log1p(df.fillna(0)), nans, starts[inv_mask], ends[inv_mask]


def lag_indexes(begin, end) -> List[pd.Series]:
    """
    # todo Series 的index是日期，值是range的结果。主程序stack后只剩range后的结果
    Calculates indexes for 3, 6, 9, 12 months backward lag for the given date range
    :param begin: start of date range
    :param end: end of date range
    :return: List of 4 Series, one for each lag. For each Series, index is date in range(begin, end), value is an index
     of target (lagged) date in a same Series. If target date is out of (begin,end) range, index is -1
    """
    dr = pd.date_range(begin, end)
    # print("begin, end",begin, end)
    # key is date, value is day index
    base_index = pd.Series(np.arange(0, len(dr)), index=dr)

    def lag(offset):
        dates = dr - offset
        # print("Series",pd.Series(data=base_index.loc[dates].fillna(-1).astype(np.int16).values, index=dr))
        # print("base_index",base_index.shape,dates.tolist(),'----\n',type(dates),type(dates.tolist()),base_index)
        # print("base_index.loc[dates.tolist()]",base_index.loc[dates])
        # todo
        #  dates (867,) <class 'pandas.core.indexes.datetimes.DatetimeIndex'> DatetimeIndex(['2015-04-01', '2015-04-02', '2015-04-03', '2015-04-04',
        #                '2015-04-05'],
        #               dtype='datetime64[ns]', freq=None)
        #  tmp1 <class 'pandas.core.series.Series'> (867,) 2015-04-01      NaN
        #  2015-04-02      NaN
        #  2015-04-03      NaN
        #                ...
        #  2017-08-12    773.0
        #  2017-08-13    774.0
        #  Length: 867, dtype: float64
        # time.sleep(0.01)
        # print("dates",np.shape(dates),type(dates),dates[:5])
        # tmp1 = base_index.reindex(dates)
        # print("tmp1",type(tmp1),np.shape(tmp1),tmp1)
        return pd.Series(data=base_index.reindex(dates).fillna(-1).astype(np.int16).values, index=dr)

    return [lag(pd.DateOffset(months=m)) for m in (3, 6, 9, 12)]


def make_page_features(pages: np.ndarray) -> pd.DataFrame:
    """
    Calculates page features (site, country, agent, etc) from urls
    :param pages: Source urls
    :return: DataFrame with features as columns and urls as index
    """
    tagged = extractor.extract(pages).set_index('page')
    # Drop useless features为何将标题列也删除？
    features: pd.DataFrame = tagged.drop(['term', 'marker'], axis=1)
    # todo
    #  tagged=                                                                    agent  ... marker
    #  page                                                                       ...
    #  !vote_en.wikipedia.org_all-access_all-agents        all-access_all-agents  ...    NaN
    #  !vote_en.wikipedia.org_all-access_spider                all-access_spider  ...    NaN
    #  !vote_en.wikipedia.org_desktop_all-agents              desktop_all-agents  ...    NaN
    #  ...                                                                   ...  ...    ...
    #  ［Alexandros］_ja.wikipedia.org_all-access_all-ag...  all-access_all-agents  ...    NaN
    #  ［Alexandros］_ja.wikipedia.org_all-access_spider         all-access_spider  ...    NaN
    #  ［Alexandros］_ja.wikipedia.org_desktop_all-agents       desktop_all-agents  ...    NaN
    #  ［Alexandros］_ja.wikipedia.org_mobile-web_all-ag...  mobile-web_all-agents  ...    NaN
    #  [145036 rows x 5 columns];(145036, 5);
    #  &
    #  features=                                                                    agent                   site country
    #  page
    #  !vote_en.wikipedia.org_all-access_all-agents        all-access_all-agents          wikipedia.org      en
    #  !vote_en.wikipedia.org_all-access_spider                all-access_spider          wikipedia.org      en
    #  !vote_en.wikipedia.org_desktop_all-agents              desktop_all-agents          wikipedia.org      en
    #  ...                                                                   ...                    ...     ...
    #  ［Alexandros］_ja.wikipedia.org_all-access_all-ag...  all-access_all-agents          wikipedia.org      ja
    #  ［Alexandros］_ja.wikipedia.org_all-access_spider         all-access_spider          wikipedia.org      ja
    #  ［Alexandros］_ja.wikipedia.org_desktop_all-agents       desktop_all-agents          wikipedia.org      ja
    #  ［Alexandros］_ja.wikipedia.org_mobile-web_all-ag...  mobile-web_all-agents          wikipedia.org      ja
    #  [145036 rows x 3 columns];(145036, 3)
    # print(f"tagged={tagged};{tagged.shape}; features={features};{features.shape}")
    return features


def uniq_page_map(pages:Collection):
    """
    todo 同一个页面的点击量会有4个来源，将同个页面不同来源的数据map到同一行
    Finds agent types (spider, desktop, mobile, all) for each unique url, i.e. groups pages by agents
    :param pages: all urls (must be presorted)
    :return: array[num_unique_urls, 4], where each column corresponds to agent type and each row corresponds to unique url.
     Value is an index of page in source pages array. If agent is missing, value is -1
    """
    import re
    result = np.full([len(pages), 4], -1, dtype=np.int32)
    pat = re.compile(
        '(.+(?:(?:wikipedia\.org)|(?:commons\.wikimedia\.org)|(?:www\.mediawiki\.org)))_([a-z_-]+?)')
    prev_page = None
    num_page = -1
    agents = {'all-access_spider': 0, 'desktop_all-agents': 1, 'mobile-web_all-agents': 2, 'all-access_all-agents': 3}
    for i, entity in enumerate(pages):
        match = pat.fullmatch(entity)
        assert match
        page = match.group(1)
        agent = match.group(2)
        # todo
        #  page=!vote_en.wikipedia.org; agent=all-access_all-agents; num_page=-1； agents[agent]=3；entity=!vote_en.wikipedia.org_all-access_all-agents
        #  page=!vote_en.wikipedia.org; agent=all-access_spider; num_page=0； agents[agent]=0；entity=!vote_en.wikipedia.org_all-access_spider
        #  page=!vote_en.wikipedia.org; agent=desktop_all-agents; num_page=0； agents[agent]=1；entity=!vote_en.wikipedia.org_desktop_all-agents
        #  page="Awaken,_My_Love!"_en.wikipedia.org; agent=all-access_all-agents; num_page=0； agents[agent]=3；entity="Awaken,_My_Love!"_en.wikipedia.org_all-access_all-agents
        #  page="Awaken,_My_Love!"_en.wikipedia.org; agent=all-access_spider; num_page=1； agents[agent]=0；entity="Awaken,_My_Love!"_en.wikipedia.org_all-access_spider
        #  page="Awaken,_My_Love!"_en.wikipedia.org; agent=desktop_all-agents; num_page=1； agents[agent]=1；entity="Awaken,_My_Love!"_en.wikipedia.org_desktop_all-agents
        #  page="European_Society_for_Clinical_Investigation"_en.wikipedia.org; agent=all-access_all-agents; num_page=1； agents[agent]=3；entity="European_Society_for_Clinical_Investigation"_en.wikipedia.org_all-access_all-agents
        #  page="European_Society_for_Clinical_Investigation"_en.wikipedia.org; agent=all-access_spider; num_page=2； agents[agent]=0；entity="European_Society_for_Clinical_Investigation"_en.wikipedia.org_all-access_spider
        #  page="European_Society_for_Clinical_Investigation"_en.wikipedia.org; agent=desktop_all-agents; num_page=2； agents[agent]=1；entity="European_Society_for_Clinical_Investigation"_en.wikipedia.org_desktop_all-agents
        #  page="Keep_me_logged_in"_extended_to_one_year_www.mediawiki.org; agent=all-access_all-agents; num_page=3； agents[agent]=3；entity="Keep_me_logged_in"_extended_to_one_year_www.mediawiki.org_all-access_all-agents
        #  page="Keep_me_logged_in"_extended_to_one_year_www.mediawiki.org; agent=all-access_spider; num_page=4； agents[agent]=0；entity="Keep_me_logged_in"_extended_to_one_year_www.mediawiki.org_all-access_spider
        #  page="Keep_me_logged_in"_extended_to_one_year_www.mediawiki.org; agent=desktop_all-agents; num_page=4； agents[agent]=1；entity="Keep_me_logged_in"_extended_to_one_year_www.mediawiki.org_desktop_all-agents
        #  page="Keep_me_logged_in"_extended_to_one_year_www.mediawiki.org; agent=mobile-web_all-agents; num_page=4； agents[agent]=2；entity="Keep_me_logged_in"_extended_to_one_year_www.mediawiki.org_mobile-web_all-agents
        # print(f"page={page}; agent={agent}; num_page={num_page}； agents[agent]={agents[agent]}；entity={entity}")
        if page != prev_page:
            prev_page = page
            num_page += 1
        result[num_page, agents[agent]] = i
    # print("result[:num_page+1]",result[:num_page+1])
    return result[:num_page+1]


def encode_page_features(df) -> Dict[str, pd.DataFrame]:
    """
    todo 对中介、和onehot编码。网页全名是index不会编码
    Applies one-hot encoding to page features and normalises result
    :param df: page features DataFrame (one column per feature)
    :return: dictionary feature_name:encoded_values. Encoded values is [n_pages,n_values] array
    """
    def encode(column) -> pd.DataFrame:
        one_hot = pd.get_dummies(df[column], drop_first=False)
        # noinspection PyUnresolvedReferences
        return (one_hot - one_hot.mean()) / one_hot.std()
    # todo
    #  df.head=                                                                    agent           site country
    #  page
    #  !vote_en.wikipedia.org_all-access_all-agents        all-access_all-agents  wikipedia.org      en
    #  !vote_en.wikipedia.org_all-access_spider                all-access_spider  wikipedia.org      en
    #  !vote_en.wikipedia.org_desktop_all-agents              desktop_all-agents  wikipedia.org      en
    #  "Awaken,_My_Love!"_en.wikipedia.org_all-access_...  all-access_all-agents  wikipedia.org      en
    #  "Awaken,_My_Love!"_en.wikipedia.org_all-access_...      all-access_spider  wikipedia.org      en
    # print(f"df.head={df.head()}")
    return {str(column): encode(column) for column in df}


def normalize(values: np.ndarray):
    return (values - values.mean()) / np.std(values)


def run():
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('data_dir')
    parser.add_argument('--valid_threshold', default=0.0, type=float, help="Series minimal length threshold (pct of data length)")
    parser.add_argument('--add_days', default=63, type=int, help="Add N days in a future for prediction")
    parser.add_argument('--start', help="Effective start date. Data before the start is dropped")
    parser.add_argument('--end', help="Effective end date. Data past the end is dropped")
    parser.add_argument('--corr_backoffset', default=0, type=int, help='Offset for correlation calculation')
    args = parser.parse_args()
    # print("args",args.start)
    # todo python make_features.py data/vars --add_days=63

    # Get the data
    df, nans, starts, ends = prepare_data(args.start, args.end, args.valid_threshold)
    # print(f"starts={starts},ends={ends}; df.head()={df.head()}")

    # Our working date range
    # todo 页面最早最晚访问时间
    # print("df", df.head(), df.shape)
    data_start, data_end = df.columns[0], df.columns[-1]

    # We have to project some date-dependent features (day of week, etc) to the future dates for prediction
    features_end = data_end + pd.Timedelta(args.add_days, unit='D')
    # todo start: 2015-07-01, end:2017-09-11 00:00:00, features_end:2017-11-13 00:00:00
    # print(f"start: {data_start}, end:{data_end}, features_end:{features_end}")

    # Group unique pages by agents
    # print("df.index.is_monotonic_increasing",df.index.is_monotonic_increasing)
    assert df.index.is_monotonic_increasing
    page_map = uniq_page_map(df.index.values)
    # print(f"page_map={page_map}")

    # Yearly(annual) autocorrelation
    raw_year_autocorr = batch_autocorr(df.values, 365, starts, ends, 1.5, args.corr_backoffset)
    # todo 相关系数为空的数量
    year_unknown_pct = np.sum(np.isnan(raw_year_autocorr))/len(raw_year_autocorr)  # type: float

    # Quarterly autocorrelation
    raw_quarter_autocorr = batch_autocorr(df.values, int(round(365.25/4)), starts, ends, 2, args.corr_backoffset)
    quarter_unknown_pct = np.sum(np.isnan(raw_quarter_autocorr)) / len(raw_quarter_autocorr)  # type: float

    print("Percent of undefined autocorr = yearly:%.3f, quarterly:%.3f" % (year_unknown_pct, quarter_unknown_pct))

    # Normalise all the things使用0代替数组x中的nan元素，使用有限的数字代替inf元素
    # todo 对年度相关系数和季度相关系数做均值方差标准化
    year_autocorr = normalize(np.nan_to_num(raw_year_autocorr))
    quarter_autocorr = normalize(np.nan_to_num(raw_quarter_autocorr))

    # Calculate and encode page features
    page_features = make_page_features(df.index.values)
    encoded_page_features = encode_page_features(page_features)
    # todo encoded_page_features 对网页的中介和国家做了 onehot 编码
    # print("encoded_page_features",type(encoded_page_features))

    # Make time-dependent features
    features_days = pd.date_range(data_start, features_end)
    #dow = normalize(features_days.dayofweek.values)
    week_period = 7 / (2 * np.pi)
    dow_norm = features_days.dayofweek.values / week_period
    dow = np.stack([np.cos(dow_norm), np.sin(dow_norm)], axis=-1)

    # Assemble indices for quarterly lagged data
    # todo 分别滑窗3，6，9，12个月的时间index
    lagged_ix = np.stack(lag_indexes(data_start, features_end), axis=-1)
    # todo data_start=2015-07-01;
    #  lagged_ix=[[ -1  -1  -1  -1]
    #  [ -1  -1  -1  -1]
    #  ...
    #  [773 681 592 500]
    #  [774 682 593 501]]
    # print(f"data_start={data_start}; lagged_ix={lagged_ix}")

    page_popularity = df.median(axis=1)
    # todo page_popularity (145036,)
    # todo 这里用每个页面的访问量的中位数表示该页面的流行程度
    # print("page_popularity",page_popularity.shape,page_popularity)
    page_popularity = (page_popularity - page_popularity.mean()) / page_popularity.std()

    # Put NaNs back
    df[nans] = np.NaN

    # Assemble final output
    # todo
    #  hits    :   过滤掉了访问量很多为0的网页，已经按照网页名称排序，并做log变换
    #  lagged_ix   :   日期index的滑窗，小于最早日期的-1做填充
    #  page_map    :   背景：每个页面最多4种中间商。数据：每一行代表一个页面名称，每一列代表一个中间商
    #  page_ix :   每个网页全名（包括页面名称，中间商，国家等）
    #  pf_agent    :   网页的中间商的onehot编码
    #  pf_country  :   网页国家的onehot编码
    #  pf_site :   网页地址sire编码
    #  page_popularity :   用页面访问量的中位数代表流行度
    #  year_autocorr   :   访问量滑窗一年的相关系数
    #  quarter_autocorr    :   访问量滑窗一个季度的相关系数
    #  dow :   周几转换为sin,cos
    tensors = dict(
        hits=df,
        lagged_ix=lagged_ix,
        page_map=page_map,
        page_ix=df.index.values,
        pf_agent=encoded_page_features['agent'],
        pf_country=encoded_page_features['country'],
        pf_site=encoded_page_features['site'],
        page_popularity=page_popularity,
        year_autocorr=year_autocorr,
        quarter_autocorr=quarter_autocorr,
        dow=dow,
    )
    # todo
    #  data_start  :访问量的起始日期
    #  data_end:访问量的终止日期
    #  features_end:特征的终止日期，比访问量的终止日期晚63天
    #  features_days:特征横跨多少天
    #  data_days:总共有多少天的数据
    #  n_pages :多少个页面
    plain = dict(
        features_days=len(features_days),
        data_days=len(df.columns),
        n_pages=len(df),
        data_start=data_start,
        data_end=data_end,
        features_end=features_end
    )
    # todo len(features_days)=867,len(df.columns)=805,len(df)=145036,data_start=2015-07-01,
    #  data_end=2017-09-11 00:00:00,features_end=2017-11-13 00:00:00
    # print(f"len(features_days)={len(features_days)},len(df.columns)={len(df.columns)},len(df)={len(df)},data_start={data_start},data_end={data_end},features_end={features_end}")
    # Store data to the disk
    VarFeeder(args.data_dir, tensors, plain)


if __name__ == '__main__':
    run()
    # prepare_data(None,None,0.0)
    print("make_feature over")
