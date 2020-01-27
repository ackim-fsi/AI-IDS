# -*- coding: utf-8 -*-
# 일반적인 검색을 위한 쿼리
# 방향성, 필터링 등 임의로 추가해서 새로 함수 작성 바랍니다.
# 패킷보다 TAS 데이터가 과도하게 짧은 경우 (2/3 미만) 제외
# 방향성 외*, src_port > dest_port 인 경우만 추출


def search_query_total(earliest_minute=-1450, latest_minute=-1440, headers=[]):
    args = {
            "earliest_minute": str(earliest_minute),
            "latest_minute": str(latest_minute),
            "earliest_notable": str(earliest_minute-10),
            "latest_notable": str(latest_minute+10),
            "headers": ' '.join(headers)
    }

    query = """
earliest={earliest_minute}m@m latest={latest_minute}m@m
index=fsi_tas_payload
`ai_hosts`
`ai_exception`
app=http
`ai_src_ip_not_korea`
NOT [ inputlookup fsi_except_ip | fields ip | rename ip as src_ip ]
NOT [ inputlookup fsi_except_ip | fields ip | rename ip as dest_ip ]
src_content!="GET / HTTP/1.1 Host: www"
src_content!="HTTP/1.*"
| append
[ search
earliest={earliest_notable}m@m latest={latest_notable}m@m
index=notable
| `get_event_id`
| rename event_id as rule_id
| fields - host
| eval suppression = if(match(eventtype, "suppression"), 1, 0)
]
| stats
earliest(_time) as _time,
values(src_content) as src_content,
values(TOP_CATE_NM) as TOP_CATE_NM,
values(MID_CATE_NM) as MID_CATE_NM,
values(msg) as msg,
values(host) as host,
values(rule_id) as rule_id,
max(suppression) as suppression
by src_ip dest_ip src_port dest_port
| search
src_content=*
AND
(
NOT msg=*
OR
(
msg!="*access*"
msg!="*admin*"
msg!="*loginpage*"
msg!="*SSH*"
msg!="*Heartbleed*"
msg!="*brute*"
msg!="smtp expn root"
msg!="F-INV-ADM-160919-MYSQL_login_attempt"
msg!="MALWARE-TOOLS Havij advanced SQL injection tool user-agent string"
msg!="FILE-*"
)
)
| where len(src_content) > 50
| lookup ip_list tas as host output cust_nm
| lookup incident_review_lookup_max_match rule_id OUTPUT anl_key
| eval anl_key = mvindex(anl_key, 0)
| eval common = "1"
| lookup fsi_analysis anl_key common OUTPUT post title
| eval post = mvindex(mvsort(post), -1)
| eval title = mvindex(title, 0)
| eval drill = if(match(title, "점검"), 1, 0)
| eval suppression = if(isnull(suppression), 0, suppression)
| eval msg = "Clean Traffic"
| eval msg = if(suppression=1, "예외처리", msg)
| eval msg = if(isnotnull(title), title, msg)

| eval label = if(drill=1, 1, 0)
| eval label = if(suppression=1, 0, label)
| eval label = if((post="2" OR post="4"), 1, label)
| eval label = if((post="1" OR post="3"), 0, label)

| table {headers}
"""
    query = query.format(**args)

    return query


def search_query_payload(earliest_minute=-205, latest_minute=-195, headers=[]):
    args = {
            "earliest_minute": str(earliest_minute),
            "latest_minute": str(latest_minute),
            "headers": ' '.join(headers)
    }

    query = """
earliest={earliest_minute}m@m latest={latest_minute}m@m
(index=notable OR eventtype=fsi_ids_event)
OR
(
index=fsi_tas_payload
`ai_hosts`
`ai_exception`
app=http
`ai_src_ip_not_korea`
NOT [ inputlookup fsi_except_ip | fields ip | rename ip as src_ip ]
NOT [ inputlookup fsi_except_ip | fields ip | rename ip as dest_ip ]
src_content!="GET / HTTP/1.1 Host: www"
src_content!="HTTP/1.*"
)
| stats
earliest(_time) as _time,
values(src_content) as src_content,
values(msg) as msg,
values(host) as tas
by src_ip dest_ip src_port dest_port
| search src_content=* src_ip!="" AND NOT msg=* 
| eval tas = mvfilter(match(tas, "TAS$"))
| where len(src_content) > 50
| table {headers}
"""
    query = query.format(**args)

    return query


def search_query_label(earliest_minute=-1450, latest_minute=-1440, headers=[]):
    args = {
            "earliest_minute": str(earliest_minute),
            "latest_minute": str(latest_minute),
            "headers": ' '.join(headers)
    }

    query = """
earliest={earliest_minute}m@m latest={latest_minute}m@m
index=ai_result
| stats latest(label) as label latest(comment) as comment by payload_id, model_name, _time, src_ip, src_port, dest_ip, dest_port, src_content
| where label > -1
| table {headers}
"""
    query = query.format(**args)

    return query



def update_notable_lookup(lookup_name, bulk_string, column_order):

    if not column_order:
        column_order = ["_time","src_ip", "src_port", "dest_ip", "dest_port", "src_content"]
    eval_query = ""
    for idx in range(len(column_order)):
        eval_query = eval_query + '''
| eval '''
        eval_query = eval_query + column_order[idx] + " = mvindex(item_list, "
        eval_query = eval_query + str(idx) + ")"

    args = {
        "bulk_string" : bulk_string,
        "eval_query" : eval_query,
        "column_order" : ' '.join(column_order),
        "lookup_name" : lookup_name
    }
    query = ''
    query = query + '''
| noop
| stats c as text
| eval text = "{bulk_string}"
| eval text = split(text, "@@row_seperator@@")
| mvexpand text
| eval item_list = split(text, "@@column_seperator@@")
{eval_query}
| table {column_order}
| eval label = ""
| eval comment = ""
| table label comment {column_order}
| append
[ inputlookup {lookup_name}
| table label comment {column_order}
]
| outputlookup {lookup_name}'''
    query = query.format(**args)

    return query



def search_query_notable_category(top_category, mid_category, earliest_minute=-1450, latest_minute=-1440):

    earliest_minute_tas = earliest_minute - 3
    latest_minute_tas = latest_minute + 2

    query = ''
    query = query + 'earliest=' + str(earliest_minute_tas) + 'm@m '
    query = query + 'latest=' + str(latest_minute_tas) + 'm@m '
    query = query + '''
index=fsi_tas_payload 
app=http
src_content=*
[
search '''
    query = query + 'earliest=' + str(earliest_minute) + 'm@m '
    query = query + 'latest=' + str(latest_minute) + 'm@m '
    query = query + 'index=notable TOP_CATE_NM="' + top_category + '" MID_CATE_NM="' + mid_category + '"'
    query = query + '''
text=*
direction=외*
msg!="*access*"
msg!="*admin*"
msg!="*loginpage*"
msg!="*SSH*"
msg!="*Heartbleed*"
msg!="*brute*"
msg!="smtp expn root"
msg!="F-INV-ADM-160919-MYSQL_login_attempt"
msg!="MALWARE-TOOLS Havij advanced SQL injection tool user-agent string"
msg!="FILE-*"
[ inputlookup correlationsearches_lookup where rule_description="침입탐지이벤트" OR rule_description="상관분석이벤트"
| fields savedsearch
| rename savedsearch as search_name
]
| search NOT `suppression`
| where src_port > dest_port
| fields src_ip src_port dest_ip dest_port
]
| fields _time src_ip src_port dest_ip dest_port src_content
| join type=inner src_ip src_port dest_ip dest_port
[
search '''
    query = query + 'earliest=' + str(earliest_minute) + 'm@m '
    query = query + 'latest=' + str(latest_minute) + 'm@m '
    query = query + 'index=notable TOP_CATE_NM="' + top_category + '" MID_CATE_NM="' + mid_category + '"'
    query = query + '''
text=*
direction=외*
msg!="*access*"
msg!="*admin*"
msg!="*loginpage*"
msg!="*SSH*"
msg!="*Heartbleed*"
msg!="*brute*"
msg!="smtp expn root"
msg!="F-INV-ADM-160919-MYSQL_login_attempt"
msg!="MALWARE-TOOLS Havij advanced SQL injection tool user-agent string"
msg!="FILE-*"
[ inputlookup correlationsearches_lookup where rule_description="침입탐지이벤트" OR rule_description="상관분석이벤트"
| fields savedsearch
| rename savedsearch as search_name
]
| search NOT `suppression`
| where src_port > dest_port
| `get_event_id`
| rename event_id as rule_id
]
| where len(src_content)*3 >= len(text)*2
| fields src_content msg rule_id _time src_ip src_port dest_ip dest_port
| lookup incident_review_lookup_max_match rule_id OUTPUT anl_key
| eval anl_key = mvindex(anl_key, 0)
| eval common = "1"
| lookup fsi_analysis anl_key common OUTPUT post title
| eval post = mvindex(post, -1)
| eval title = mvindex(title, -1)
| eval label = -1
| eval label = if((post="2" OR post="4" OR match(title, "점검")), 1, label)
| eval label = if((post="1" OR post="3"), 0, label)
| search label != -1
| table src_content msg label _time src_ip src_port dest_ip dest_port
'''
    return query


def search_query_notable_category_test(top_category, mid_category, earliest_minute=-1450, latest_minute=-1440):

    latest_notable = latest_minute + 240
    if latest_notable > 0:
        latest_notable = 0
    args = {
            "earliest_minute": str(earliest_minute),
            "latest_minute": str(latest_minute),
            "latest_notable": str(latest_notable),
            "top_category": top_category,
            "mid_category": mid_category
        }
    query = """
earliest={earliest_minute}m@m latest={latest_notable}m@m
index=ai_kvstore source=fsi_analysis approve=1
| fields title post stix anl_key
| stats values(stix) as stix, last(title) as title, max(post) as post by anl_key
| mvexpand stix
| search post!=1
| eval label = if(post=2 OR post=4, 1, if(match(title, "점검"), 1, 0))
| fields stix label
| eval stix = split(trim(stix, "|^^|"), "|^^|")
| mvexpand stix
| search stix = *
| eval notable_fields = split(stix, "|^|")
| eval src_ip = mvindex(notable_fields, 0)
| eval src_port = mvindex(notable_fields, 1)
| eval dest_ip = mvindex(notable_fields, 3)
| eval dest_port = mvindex(notable_fields, 4)
| eval top_category = mvindex(notable_fields, 6)
| eval mid_category = mvindex(notable_fields, 7)
| eval msg = mvindex(notable_fields, 8)
| eval time = strptime(mvindex(notable_fields, 19), "%Y-%m-%d %H:%M:%S")
| eval time_start = floor(now()/60)*60 + 60*({earliest_minute})
| eval time_end = floor(now()/60)*60 + 60*({latest_minute})
| eval in_time = if(time_start<=time and time<=time_end, 1, 0)
| fields - stix notable_fields
| search
in_time=1
top_category="{top_category}"
mid_category="{mid_category}"
msg!="*access*"
msg!="*admin*"
msg!="*loginpage*"
msg!="*SSH*"
msg!="*Heartbleed*"
msg!="*brute*"
msg!="smtp expn root"
msg!="F-INV-ADM-160919-MYSQL_login_attempt"
msg!="MALWARE-TOOLS Havij advanced SQL injection tool user-agent string"
msg!="FILE-*"
| join type=inner src_ip src_port dest_ip dest_port
[ search
earliest={earliest_minute}m@m latest={latest_minute}m@m
index=ai_tas_payload source=stream:ai*
`ai_field`
| search src_content=*
| eval time = _time
| fields time src_ip src_port dest_ip dest_port src_content
]
| eval _time = mvindex(time, 0)
| table _time label src_ip src_port dest_ip dest_port src_content
"""
    query = query.format(**args)

    return query


def search_query_payload_test(earliest_minute=-1450, latest_minute=-1440):

    args = {
            "earliest_minute": str(earliest_minute),
            "latest_minute": str(latest_minute)
    }

    query = """
earliest={earliest_minute}m@m latest={latest_minute}m@m
((index=ai_tas_payload app=http) OR index=ai_notable)
NOT ahnlab
NOT microsoft
src_content!="GET / HTTP/1.1 Host: www"
`ai_src_ip_not_korea`
`ai_field`
| stats dc(index) as dc_index, earliest(_time) as _time, values(index) as values_index, values(src_content) as src_content by src_ip dest_ip src_port dest_port
| search dc_index=1 values_index="ai_tas_payload" src_content!=HTTP/1.*
| fields src_content src_ip dest_ip src_port dest_port 
| eval msg = "Clean Traffic"
| eval label = 0
| fields _time src_content label src_ip src_port dest_ip dest_port
| table _time label src_ip src_port dest_ip dest_port src_content 
"""
    query = query.format(**args)

    return query

