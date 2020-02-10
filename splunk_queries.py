# -*- coding: utf-8 -*-
# general search for splunk queries


def search_query_total(earliest_minute=-1450, latest_minute=-1440, headers=[]):
    args = {
            "earliest_minute": str(earliest_minute),
            "latest_minute": str(latest_minute),
            "earliest_notable": str(earliest_minute-10),
            "latest_notable": str(latest_minute+10),
            "headers": ' '.join(headers)
    }

    query = """
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

[
search '''
    query = query + 'earliest=' + str(earliest_minute) + 'm@m '
    query = query + 'latest=' + str(latest_minute) + 'm@m '
    query = query + 'index=notable TOP_CATE_NM="' + top_category + '" MID_CATE_NM="' + mid_category + '"'
    query = query + '''

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

"""
    query = query.format(**args)

    return query


def search_query_payload_test(earliest_minute=-1450, latest_minute=-1440):

    args = {
            "earliest_minute": str(earliest_minute),
            "latest_minute": str(latest_minute)
    }

    query = """
    
"""
    query = query.format(**args)

    return query

