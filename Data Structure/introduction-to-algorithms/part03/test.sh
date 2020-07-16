#------------------------------------------------
# OUT:
#   test
#
# DEPS:
#   
#
# DEV: meiyu.wen
#
#
# DESC: 
#
#
# DATE: 20200210
#------------------------------------------------

function rpt_gmv_forecast_fact_compare_run() {

  init && \
  rpt_gmv_forecast_fact_compare && \
  clean_tmp_data

}

function init() {

  DT_F="$(date -d "${DATE}" "+%Y-%m-%d")"
  TMP_DATA_PATH="/tmp/dm_gmv_forecast_fact_compare_${DATE}.csv"

  FILE_HTML="html_gmv_forecast_fact_compare_${DATE}.txt"
  STR_COLUMNS="月份,渠道,日均GMV,新客日均GMV,老客日均GMV,月目标,当前达成,达成进度"
  IS_COLUMNS="False"
  TMP_PATH="/tmp"
  COMMENT="GMV 完成进度"

  m_mail_title='GMV 完成进度报表'
  m_receivers='wenmeiyu@czb365.com,842327356@qq.com,mawanlong@czb365.com'

  hive_base_settings="
    set mapred.child.java.opts=-Xmx4096m;
    set mapreduce.reduce.memory.mb=4096;
    set mapreduce.reduce.java.opts='-Xmx4096M';
    set mapreduce.map.memory.mb=1000;
    set mapreduce.map.java.opts='-Xmx3600M';
    set mapred.child.map.java.opts='-Xmx3600M';
    set mapred.job.priority=HIGHT;
    set hive.exec.compress.output=true;
    set hive.optimize.reducededuplication=false;
    set mapred.reduce.tasks = 2;
  "
}