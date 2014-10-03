#!/usr/bin/env python
"""
LiveMR returns a JSON like dict of the imput 30 minute stream:

base chunks:
base_user[uname] = [events: "", times: ""]
base_videos[url] = [base_user,base_user]

clients[client] = {videos= {users={[name:"",events:"",times:""]}}}
"""

import json
import sys
import urlparse
import datetime
import time
import csv
import numpy as np

from disco.core import Job, result_iterator
import disco
import disco.ddfs
import pdb
import re

class LiveMR(disco.core.Job):
    def map(self,line,params):
        row = json.loads(line)
        # line = "client timestamp ip video user event extra TYPE"
        client = ''.join(("%s" % row.get('clientKey')).split(' '))

        line = [client, row.get('timestamp'), ''.join(row.get('clientIp','none').split(' '))] 
        if row.get('query','none') != 'none':
            #video = ''.join(("%s" % row.get('clientKey')).split(' '))
            line.append(row.get('query').get('u','none'))
            if row.get('query').get('uid','none') != '0':
                line.append(row.get('query').get('uid','none')) 
            else: 
                line.append(row.get('clientIp'))
            line.append(row.get('query').get('a','none'))
            try: line.append(''.join(row.get('query').get('l','none').encode('ascii').split(' ')))
            except: 
                line.append(row.get('query').get('l','none'))
            line.append(row.get('query').get('c','none'))
            line.append(row.get('query').get('r','none'))
            line.append(row.get('query').get('s','none'))
        else:
            line.append("none none none none none")
        yield ("%s %i %s %s %s %s %s %s %s" % (line[0],line[1],line[2],line[3],line[4],line[5],line[6],line[8],line[9]), ((line[7]=='video')&(line[0]!='None')&('' not in line)&(line[4] !='0')))

    @staticmethod
    def go(input):
        job = LiveMR()
        job.run(input=input,
            map_reader=disco.worker.classic.func.chain_reader,
            partitions=1)
            #combiner=DailyClientActivity.combiner

        top_list = list(result_iterator(job.wait(show=False)))
        for url, use in top_list:
            if use: yield url

if __name__ == '__main__':
    from LiveMR import LiveMR
    run_with = disco.ddfs.DDFS().list(prefix="events:time")[-4:]
    clients = {}
    t0 = time.time()
    debug = False
    print run_with
    for line in LiveMR.go(run_with):
        try: 
            client,timestamp,ip,video,user,event,extra = line.split()
        except:
            if debug:
                print "FAILED::"
                print "   ",line
                pdb.set_trace()
            #pdb.set_trace() 
        #print client,timestamp,video,user,event,extra
        if client not in clients:
            clients[client]={}
        if video not in clients[client]:
            clients[client][video]={}
        if user not in clients[client][video]:
            clients[client][video][user]=[[[event],[extra],[int(timestamp)]],ip]
        else:
            clients[client][video][user][0][0].append(event)
            clients[client][video][user][0][1].append(extra)
            clients[client][video][user][0][2].append(float(timestamp))
    print "finished read in %0.2f" % (time.time()-t0)
    t0=time.time()
    
    for client in clients:
        if debug: print client
        for video in clients[client]:
            if debug: print " ",video
            for user in clients[client][video]:
               
                events   = clients[client][video][user][0][0]
                progress = clients[client][video][user][0][1]
                times    = clients[client][video][user][0][2]
                interaction = np.sort(np.array(zip(events,progress,times),dtype=[('events','object'),('progress','object'),('times','float')]),order='times')
                #for int in interaction: print int

                if ('play' in interaction['events']) and ('progress' in interaction['events']):
                    ## "interaction score"
                    #pdb.set_trace()
                    Tplay = np.min(interaction[interaction['events']=='play']['times'])
                    Ninteract =  len((','.join([prog for prog in interaction[interaction['events']=='progress']['progress']])).split(','))
                    interact_times = [float(t) for t in re.split(',|-',('-'.join([prog for prog in interaction[interaction['events']=='progress']['progress']])))]
                   
                    Tpredict =  max(interact_times)
                    #Tinteract_predict =  interaction[interaction['events']=='progress']['times'][-1]-Tplay #np.max(interact_times)-np.min(interact_times)
                    Tinteract = np.max(interaction[interaction['events']=='progress']['times'])-Tplay#np.min(interaction[interaction['events']=='progress']['times'])
                 
                    ## "ADD score"
                    add = interaction[interaction['events']=='progress']
                    add_arr = np.array([])
                    spans = add['times']-np.append(Tplay,add['times'][:-1])
                    for i in np.arange(len(add)):
                        for span in add[i]['progress'].split(','): 
                            add_arr = np.append(add_arr,(float(span.split('-')[1])-float(span.split('-')[0]))/spans[i])
                    add_arr =add_arr[np.isfinite(add_arr)]
                    add_score = (1.0-np.min([1.0,np.sum(add_arr)/len(add_arr)]))/0.9
                else:
                    #print "no user interaction!"
                    Tinteract = -1.
                    Tpredict = -1.
                    add_score = -1.
 
                if debug: print "  Tint=%0.3f  Tscore=%0.3f   ADD score=%0.3f  user=%s\n" % (Tinteract,Tinteract/Tpredict,add_score,user)
                clients[client][video][user] = [Tinteract,Tpredict,add_score,clients[client][video][user][1]]
            #print ""
        if debug: pdb.set_trace()
                #print events
    if debug: pdb.set_trace() 
    print "finished minimization in %0.2f" % (time.time()-t0)

    pdb.set_trace()

