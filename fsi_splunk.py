# -*- coding: utf-8 -*-
import requests
import time
import re
import xml.etree.ElementTree as et


#########################################################
### Splunk Query Setup                                  #
### search_query (string) : Query                     #
### period (int) : duration for task completion (sec)        #
### output_format (string) : outputfile format (csv, xml, json) #
### auth (string tuple) : (ID, PW)                       #
### output_count (int) : Number of ouput count          #
### output file format : csv, xml, json                 #
### return (string) : result of query                   #
#########################################################
def query(splunk_host, search_query, check_frequency, output_format, auth, sample_ratio=1, output_count=0):

    if not search_query.startswith('|'):

        if 'latest' not in search_query:
            search_query = 'latest=now ' + search_query

        if 'earliest' not in search_query:
            search_query = 'earliest=-15m@m ' + search_query

        if not search_query.startswith('search'):
            search_query = 'search ' + search_query


    if output_format not in ['csv', 'xml','json']:
        return ''

    splunk_job_url = splunk_host + "/services/search/jobs"
    search_response = requests.post(splunk_job_url,
                                    data = {'search':search_query,
                                            'dispatch.sample_ratio':sample_ratio},
                                    auth = auth,
                                    verify=False)
    # Job has been submitted.
    try:
        search_root = et.fromstring(search_response.text)
        splunk_sid = search_root.find('sid').text
    except AttributeError:
        print(search_response.text)
        exit(0)
        return None
    except et.ParseError:
        print(search_response.text)
        exit(0)

    while True:
        time.sleep(check_frequency)
        job_response = requests.get(splunk_job_url + '/' + splunk_sid,
                                    auth = auth,
                                    verify=False)

        job_status = re.search('<s:key name="dispatchState">(.+)</s:key>', job_response.text).group(1)

        #job_root = et.fromstring(job_response.text)
        #ns = {'atom': 'http://www.w3.org/2005/Atom', 's': 'http://dev.splunk.com/ns/rest'}
        #job_status = job_root.find("./atom:content/s:dict/s:key[@name='dispatchState']", ns).text

        if job_status == 'DONE': # Job is finished
            break
        if job_status == 'FAILED': # Job is finished
            print('Search Failed!')
            print(job_response.text)
            exit(0)


    splunk_result = requests.get(splunk_job_url + '/' + splunk_sid
                                 + '/results?output_mode=' + output_format
                                 + '&count=' + str(output_count),
                                 auth = auth,
                                 verify=False)

    if '<msg type="' in splunk_result.text and '<messages>' in splunk_result.text:
        print(splunk_result.text)
        return None

    return splunk_result.text.strip('\n')
