"""
mediaRecommend.py

Constructs video recommendation model for embedly clients.

The model is constructed using the way that users are interacting 
with content on a site.  

written by zgazak, Sept 2014
"""

### Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pylab as pylab

### general stats and array manips
import pandas as pd
import numpy as np
import math
#from collections import defaultdict12

### basic information and system interaction
import re
import os
import sys
import copy
import urllib
import logging
from time import time

### file IO
import json
from pprint import pprint
import csv

### Machine Learning
### Latent Dirichlet Allocation
from gensim import corpora, models, similarities

### Imports for disco and embedly contexts
import urlbin
import requests
import disco.ddfs
import regan

### LiveMR is a home grown clunky Map/Reduce program
from LiveMR import LiveMR

### debugging
import pdb

def r_callback(response):
  if response.status_code > 299:
    print "Failure"

class mediaRecommend(object):

    def __init__(self,debug=False):
        self.version = 'v1.1'
        self.debug = debug


    """ controller function for building a model across a domain
    """
    def build_domain_model(self,domainlist=None,max_chunks=24,max_vids=10000,nrec=10):
        self.runMode = 'domain'
        tick = time()
        self.max_chunks = max_chunks
        self.max_vids = max_vids
        self.nrec = nrec
        ### step 1: acquire data...    
        ## compiles input into a numpy array: self.data_array
        self.domainlist = domainlist
        self.load_data_domain(domainlist=domainlist[:])  

        for domain in self.domainlist:
            self.domainKey = domain
            self.data_dict = self.domain_dict[domain]
            if len(self.data_dict) > self.max_vids:
                short = [key for key in self.data_dict.keys() if len(self.data_dict[key]) == 1]
                for key in short: del self.data_dict[key]
            if self.debug:
                print "### building recommendation model for %s." % (self.domainKey)
                t0 = time()

            ### step 2: build the user interaction document dictionary
            ### constructs self.documents[video]=[np.array([user,interact_score,concentration_score])]
            self.build_variables()

            ### step 4: Construct Latent Dirichlet Allocation + cultivated scalars model
            self.build_LDA(pop_weight=0.0,i_weight=0.5,f_weight=0.5)
            
            ### step 5: distibute model to Tornado MediaRecommend database
            self.distribute_LDA(load_to_tornado=False)

            if self.debug:
                print "### domain recommendation model complete in %0.1fs" % (time()-t0)
        print "completed model building loop in %0.2f minutes" % ((time()-tick)/60)

    """ controller function for building a model for a client site.
    """
    def build_client_model(self,clientlist=None,max_chunks=24,max_vids=10000,nrec=10):
        self.runMode = 'client'
        tick = time()
        self.max_chunks = max_chunks
        self.max_vids = max_vids
        self.nrec = nrec

        ### step 1: acquire data...    
        ## compiles input into a numpy array: self.data_array
        self.clientlist = clientlist
        self.load_data(clientlist=clientlist[:])  

        for client in self.clientlist:
            self.clientKey = client
            self.data_dict = self.client_dict[client]
            if len(self.data_dict) > self.max_vids:
                short = [key for key in self.data_dict.keys() if len(self.data_dict[key]) == 1]
                for key in short: del self.data_dict[key]
            if self.debug:
                print "### building recommendation model for %s." % (self.clientKey)
                t0 = time()

            ### step 2: build the user interaction document dictionary
            ### constructs self.documents[video]=[np.array([user,interact_score,concentration_score])]
            self.build_variables()

            ### step 4: Construct Latent Dirichlet Allocation + cultivated scalars model
            self.build_LDA(pop_weight=0.4,i_weight=0.2,f_weight=0.2)

            ### step 5: distibute model to Tornado MediaRecommend
            self.distribute_LDA(load_to_tornado=True)

            ### step 6: export LDA for visualization
            ### for testing only
            # self.save_LDA()

            if self.debug:
                print "### client recommendation model complete in %0.1fs" % (time()-t0)
        print "completed model building loop in %0.2f minutes" % ((time()-tick)/60)

    def build_LDA(self,pop_weight=1.0,i_weight=1.0,f_weight=1.0):
        if self.debug:
            print "building LDA topics..."
            tock = time()

        LDA_data = pd.DataFrame([self.documents[video] for video in self.documents],columns=['documents','users','interacts','focuses','vid_length','focus','iscore','popularity','ip','embed_url','domain'])
        LDA_data['iscore_store'] = LDA_data['iscore']
        LDA_data['urls'] = [video for video in self.documents] 
        LDA_data = LDA_data[LDA_data['iscore'] > 0]
        if np.min(LDA_data['iscore']) < 0: LDA_data['iscore']-=np.min(LDA_data['iscore'])
        LDA_data['iscore'] = [np.min([1,f]) for f in LDA_data['iscore']]

        ## scale popularity to remove outliers... max popularity is the minimum popularity of the top 1% of videos
        maxpop = np.sort(LDA_data['popularity'])[-len(LDA_data['popularity'])*.05]
        LDA_data['pop_store'] = LDA_data['popularity']/max(LDA_data['popularity'])
        LDA_data['popularity'] = [np.min([1.,float(pop)/maxpop]) for pop in LDA_data['popularity']]
        LDA_data['focus']-=np.min(LDA_data['focus'])
        LDA_data['focus']/=max(LDA_data['focus'])

        if len(LDA_data) > 10000:
            print "limit to 10000 videos of highest pop..."
            LDA_data = LDA_data.sort(['popularity'])[-10000:]

        ## calculate number of clusters
        nclust = min([150,max([5,int(len(LDA_data)/30.)])])
        if self.debug:
            print "   ... clustering %i videos into %i clusters." % (len(LDA_data),nclust)
            print "       ... build vectorizer ...",
            t0 = time()

        ## Tokenize:
        texts = [[word for word in document.lower().split()] for document in LDA_data['documents']]

        ## build "user language" dictionary
        dictionary = corpora.Dictionary(texts)

        ## make a sparse vectorization of the documents and store as the "corpus"
        corpus = [dictionary.doc2bow(word) for word in texts]

        ### utilize term-frequency inverse-document-frequency since how long someone stays on a video is important
        tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
        corpus_tfidf = tfidf[corpus] ## apply tfidf to corpus

        ## compute a latent dirichlet allocation (LDA) to design the X topics (X = nclust)
        model = models.ldamodel.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=nclust, passes=10)

        if self.debug:
            print " done in %0.1fs" % (time() - t0)
            print "       ... fitting clusters...",
            t0 = time()

        ## Try to make this more elegant... structure of model[] output is sparse... use that for speed
        clust = np.zeros((nclust,len(texts)))
        for j in np.arange(len(corpus)):
            res = model[corpus[j]]
            for i,val in res:
                try: clust[i][j] = val
                except:
                    if self.debug: 
                        pdb.set_trace()

        if self.debug:
            print " done in %0.1fs" % (time() - t0)
            print "       ... calculate cross probabilities ...",
            t0 = time()

        ### first build topic cross sections
        topics = pd.DataFrame(clust[0],columns=['c1']) #,columns=[('c%i'%(c+1)) for c in np.arange(nclust)])
        for i in np.arange(nclust-1)+1: topics[('c%i'%(i+1))]= [c for c in clust[i]]
        cross_topic_probs = np.dot(topics,topics.T)
        cross_topic_probs/=np.max(cross_topic_probs)

        ### scale those results by cultivated parameters 
        crossprob = cross_topic_probs*(np.array(LDA_data['popularity'])**pop_weight)*(np.array(LDA_data['iscore'])**i_weight)*(np.array(LDA_data['focus'])**f_weight)
        crossprob[np.isnan(crossprob)] = 0.01
        ### make sure no video is recommended itself
        crossprob[np.diag_indices(len(topics))] = 0.0
        ### recs per video

        self.recommendation_model = {}

        self.LDA_data = LDA_data
        self.cross_probabilities = crossprob
        if self.debug: 
            print " done in %0.1fs" % (time() - t0)
            print " finished in %0.1f" % (time() -tock)
            #for i in np.arange(nclust): LDA_data[('c%i'%(i+1))]= [c for c in clust[i]]
            #LDA_data.to_pickle('save_LDA_%s.sav'%(self.clientKey))
            #try:
            #    wcsv = csv.writer(open("save_recmodel_%s.csv.sav"%(self.clientKey), "w"))
            #    for key, val in self.recommendation_model.items(): wcsv.writerow([key, val])
            #    wcsv = csv.writer(open("save_crossprob_%s.csv.sav"%(self.clientKey), "w"))
            #    for line in crossprob: wcsv.writerow(line)
            #except: pdb.set_trace()

    def save_LDA(self):
        if self.debug: print "saving LDA"
        #LDA_data.to_pickle('save_LDA_%s.sav'%(self.clientKey))
        if self.runMode == 'client':
            try: self.LDA_data.to_csv('/mnt/work/save_LDA_%s_%s.csv.sav'%(self.clientKey,self.runMode))
            except: pdb.set_trace()
        if self.runMode == 'domain':
            self.LDA_data.to_csv('/mnt/work/save_LDA_%s_%s.csv.sav'%(self.domainKey,self.runMode))

    def build_key(self,search,encode=True):
        if encode: search = urlbin.Url.parse(search).key
        if self.runMode == 'client':
            return "%s_%s"%(self.clientKey,search)
        if self.runMode == 'domain':
            return "%s_%s"%(self.domainKey,search)

    def distribute_LDA(self,load_to_tornado=True):
        if self.debug:
            tock = time()
            print "...Distributing LDA model"
      
        r_tornado = regan.Regan(host='http://localhost:7346')

        ##MediaRecommend
        for (url,line) in zip(self.LDA_data['urls'],self.cross_probabilities):
            key = self.build_key(url)
            prob_matrix = np.sort(np.array(zip(line,self.LDA_data['urls'],self.LDA_data['embed_url']),dtype=[('probability','float'),('recommendation','object'),('rec_embed','object')]),order='probability')[-self.nrec*4:] 
            prob_matrix = prob_matrix[~np.isnan(prob_matrix['probability'])]
            weights = [i[0] for i in prob_matrix]
            if max(weights) == 0:
                trending_rec =  self.LDA_data.sort(columns='popularity')[-self.nrec*4:]
                prob_matrix = np.array(zip(trending_rec['popularity']/np.sum(trending_rec['popularity']),trending_rec['urls'],trending_rec['embed_url']),dtype=[('probability','float'),('recommendation','object'),('rec_embed','object')])
                prob_matrix = np.sort(np.random.choice(prob_matrix,self.nrec,replace=False),order='probability')
            else: prob_matrix = np.sort(np.random.choice(prob_matrix,self.nrec,replace=False,p=weights/np.sum(weights)),order='probability')
            try:
                self.recommendation_model[key] = {"base_url": url, "rec_embed":prob_matrix['rec_embed'] ,"recommendation": prob_matrix['recommendation'], "weight": prob_matrix['probability']/np.sum(prob_matrix['probability'])}
                #if math.isnan(np.sum(prob_matrix['probability'])): pdb.set_trace()
                r_tornado.put('MediaRecommend', key, dict(referrer_urls=["%s"%s for s in prob_matrix['rec_embed']],
                                                        urls=["%s"%s for s in prob_matrix['recommendation']], 
                                                        weights=["%0.3f"%s for s in prob_matrix['probability']/np.sum(prob_matrix['probability'])],
                                                        timestamp=time(),
                                                        request_url=url),
                                                        callback=r_callback)
            except:
                if self.debug: 
                    print "something failed for key %s" % (key)
                    #pdb.set_trace()

        ### add trending videos
        key = self.build_key('nomatch',encode=False) 
        trending_rec =  self.LDA_data.sort(columns='popularity')[-self.nrec*2:]
        prob_matrix = np.array(zip(trending_rec['popularity']/np.sum(trending_rec['popularity']),trending_rec['urls'],trending_rec['embed_url']),dtype=[('probability','float'),('recommendation','object'),('rec_embed','object')])
        r_tornado.put('MediaRecommend', key, dict(referrer_urls=["%s"%s for s in prob_matrix['rec_embed']],
                                                    urls=["%s"%s for s in prob_matrix['recommendation']], 
                                                    weights=["%0.3f"%s for s in prob_matrix['probability']/np.sum(prob_matrix['probability'])],
                                                    timestamp=time(),
                                                    request_url=url),
                                                    callback=r_callback)
        self.recommendation_model[key] = {"base_url":'nomatch', "rec_embed":prob_matrix['rec_embed'], "recommendation": prob_matrix['recommendation'], "weight": prob_matrix['probability']/np.sum(prob_matrix['probability'])}
        if self.debug:
            print key 
            print self.recommendation_model[key]

        if load_to_tornado: 
            if self.debug: 
                t0 = time()
                print "pushing to tornado."
            r_tornado.go()   
            if self.debug: print " done in %0.1fs" % (time() - tock)

        #else: pdb.set_trace()

    def build_variables(self):
        if self.debug:
            print "building variables...",
            t0 = time()

        self.documents = {}
        
        for video in self.data_dict:
            for user in self.data_dict[video]:
                events     = self.data_dict[video][user][0][0]
                prog_start = self.data_dict[video][user][0][1]
                prog_end   = self.data_dict[video][user][0][2]
                times      = self.data_dict[video][user][0][3]
                ip         = self.data_dict[video][user][1]
                embed_url  = self.data_dict[video][user][2]
                domain     = self.data_dict[video][user][3]
                interaction = np.sort(np.array(zip(events,prog_start,prog_end,times),dtype=[('event','object'),('process_start','object'),('process_end','object'),('timestamp','float')]),order='timestamp')

                if ('play' in interaction['event']) and ('progress' in interaction['event']):
                    Tplay = np.min(interaction[interaction['event']=='play']['timestamp'])
                    Tpredict = np.max([float(t) for t in interaction[np.where(interaction['event']=='progress')]['process_end']])
                    Tinteract = np.max([np.max(interaction[np.where(interaction['event']=='progress')]['timestamp'])-Tplay,
                                        np.max(interaction[np.where(interaction['event']=='progress')]['timestamp'])-np.min(interaction[np.where(interaction['event']=='progress')]['timestamp'])])
                    ### 
                    focus = interaction[interaction['event']=='progress']
                    focus_arr = np.zeros(0)
                    spans = focus['timestamp']-np.append(Tplay,focus['timestamp'][:-1])
                    time_on_vid = np.array([float(focus[i]['process_end'])-float(focus[i]['process_start']) for i in np.arange(len(focus))])
                  
                    time_on_vid = time_on_vid[np.where(spans >= 0)]
                    spans = spans[np.where(spans >= 0)]

                    if (len(spans) != 0) and (np.max(spans) > 0):
                        while (spans[0] == 0) & (len(spans) > 2): 
                            spans = spans[1:]
                            time_on_vid = time_on_vid[1:]
                        if np.min(spans) == 0: 
                            for i in np.where(spans == 0)[0]:spans[i] = spans[i-1]
                        try: focus_score = np.min([1.0,np.sum(time_on_vid/spans)/len(spans)])
                        except: 
                            if self.debug: 
                               pdb.set_trace()
                        #if pd.isnull(focus_score): pdb.set_trace()

                        if video not in self.documents:          
                            self.documents[video] = [[],[],[],[],0,0,0,0,[],'',''] ## the document [[document],[users],[Tinteracts],[focus_scores], best_guess_length, focus_avg, iscore_avg, site_pop,[ips],embed_url,domain]
                            self.documents[video][9] = embed_url
                            self.documents[video][10] = domain
                        self.documents[video][1].append(user)
                        self.documents[video][2].append(Tinteract)
                        self.documents[video][3].append(focus_score)
                        self.documents[video][4] = max([Tpredict,self.documents[video][4]])
                        self.documents[video][8].append(ip)    

            ### some videos won't make it in because no actual plays
            if video in self.documents:
                ### now compile all the stats for that video ...
                isig = np.std(self.documents[video][2]/self.documents[video][4])
                self.documents[video][6] = np.average(self.documents[video][2]/self.documents[video][4])
                self.documents[video][5] = np.average(self.documents[video][3])
                
                ## redefine popularity... total time engagement with video...
                self.documents[video][7] = np.sum(self.documents[video][2])
                ##self.documents[video][7] = len(self.documents[video][1])

                """ new recipe:
                    iscore << median (2 sig? normal assumed), negative sentiment, append user with "n"
                    iscore < median: in once
                    iscore >= median: in twice
                """
                for (user,interact) in zip(self.documents[video][1],self.documents[video][2]):
                    iscore =  (interact/self.documents[video][4]- self.documents[video][6])/isig
                    ## negative sentiment::  
                    if iscore <= -1.5: self.documents[video][0].append('neg_'+user)
                    ## general positive sentiment:: 
                    elif iscore < 1: self.documents[video][0].append(user)
                    ## significant positive sentiment:: 
                    else: 
                        for i in np.arange(min(5,round(iscore))): self.documents[video][0].append(user)

                self.documents[video][0] = ' '.join(self.documents[video][0])

        if self.debug: print " finished in %0.1f" % (time()-t0)

    def load_data_domain(self,filelist=None,domainlist=None):
        if self.debug:
            print "loading data from disco ddfs... no more than %i hours"%(self.max_chunks/2)
            t0 = time()


        ## This will be basically the same as load_data but client = domain... 

        ## which DDFS files to parse
        if filelist==None: filelist = sorted(disco.ddfs.DDFS().list(prefix="events:time"))[-self.max_chunks:]

        self.domain_dict={}
        LoadMoreHistory = True
        for filen in filelist[::-1]:
            if LoadMoreHistory:
                if self.debug: print "....%s" % (filen)
                for line in LiveMR.go([filen]):
                    try: 
                        client,timestamp,ip,video,user,event,extra,embed_url,domain = line.split()
                    except:
                        try: 
                            client,timestamp,ip,video,user,event,extra,extra2,embed_url,domain = line.split()
                            extra = ''.join([extra[:-1],']'])
                        except:
                            if self.debug:
                                print "FAILED::",line
                                #pdb.set_trace()

                    if domain in domainlist:
                        if domain not in self.domain_dict: self.domain_dict[domain] = {}
                        if video not in self.domain_dict[domain]: self.domain_dict[domain][video] = {}
                        if user  not in self.domain_dict[domain][video]: self.domain_dict[domain][video][user] = [[[],[],[],[]],ip,embed_url,client] ## [[[event],[start],[end],[timestamp]],ip]

                        if event == 'progress':
                            split = re.split('-|,',extra)
                            for i in xrange(0,len(split),2): 
                                if ((split[i] != 'NaN') and (split[i+1]!='NaN')):
                                    self.domain_dict[domain][video][user][0][0].append(event)
                                    self.domain_dict[domain][video][user][0][1].append(split[i])
                                    self.domain_dict[domain][video][user][0][2].append(split[i+1])
                                    self.domain_dict[domain][video][user][0][3].append(float(timestamp))
                        else:
                            self.domain_dict[domain][video][user][0][0].append(event)
                            self.domain_dict[domain][video][user][0][1].append(None)
                            self.domain_dict[domain][video][user][0][2].append(None)
                            self.domain_dict[domain][video][user][0][3].append(float(timestamp))
                ### continue loading?  look for 15000 videos max, then drop to 10000 most active videos (if necessary)... 
                LoadMoreHistory = False
                for dom in domainlist:
                    if len([key for key in self.domain_dict[dom] if len(self.domain_dict[dom][key]) > 1]) < self.max_vids*1.2: 
                        LoadMoreHistory=True
                    else: 
                        if self.debug: print "domain %s is done with %i videos loaded." % (dom,len([key for key in self.domain_dict[dom] if len(self.domain_dict[dom][key]) > 1]))
                        domainlist.remove(dom) 
                if len(domainlist) == 0: LoadMoreHistory = False
    
        if self.debug: print "finished read in %0.1fs." % (time()-t0)

    def load_data(self,filelist=None,clientlist=None):
        if self.debug:
            print "loading data from disco ddfs... no more than %i hours"%(self.max_chunks/2)
            t0 = time()

        ## which DDFS files to parse
        if filelist==None: filelist = sorted(disco.ddfs.DDFS().list(prefix="events:time"))[-self.max_chunks:]

        self.client_dict={}
        LoadMoreHistory = True
        for filen in filelist[::-1]:
            if LoadMoreHistory:
                if self.debug: print "....%s" % (filen)
                for line in LiveMR.go([filen]):
                    try: 
                        client,timestamp,ip,video,user,event,extra,embed_url,domain = line.split()
                    except:
                        try: 
                            client,timestamp,ip,video,user,event,extra,extra2,embed_url,domain = line.split()
                            extra = ''.join([extra[:-1],']'])
                        except:
                            if self.debug:
                                print "FAILED::",line
                                #pdb.set_trace()

                    if client in clientlist:
                        if client not in self.client_dict: self.client_dict[client] = {}
                        if video not in self.client_dict[client]: self.client_dict[client][video] = {}
                        if user  not in self.client_dict[client][video]: self.client_dict[client][video][user] = [[[],[],[],[]],ip,embed_url,domain] ## [[[event],[start],[end],[timestamp]],ip]

                        if event == 'progress':
                            split = re.split('-|,',extra)
                            for i in xrange(0,len(split),2): 
                                if ((split[i] != 'NaN') and (split[i+1]!='NaN')):
                                    self.client_dict[client][video][user][0][0].append(event)
                                    self.client_dict[client][video][user][0][1].append(split[i])
                                    self.client_dict[client][video][user][0][2].append(split[i+1])
                                    self.client_dict[client][video][user][0][3].append(float(timestamp))
                        else:
                            self.client_dict[client][video][user][0][0].append(event)
                            self.client_dict[client][video][user][0][1].append(None)
                            self.client_dict[client][video][user][0][2].append(None)
                            self.client_dict[client][video][user][0][3].append(float(timestamp))
                ### continue loading?  look for 15000 videos max, then drop to 10000 most active videos (if necessary)... 
                LoadMoreHistory = False
                for cli in clientlist:
                    if len([key for key in self.client_dict[cli] if len(self.client_dict[cli][key]) > 1]) < self.max_vids*1.2: 
                        LoadMoreHistory=True
                    else: 
                        if self.debug: print "client %s is done with %i videos loaded." % (cli,len([key for key in self.client_dict[cli] if len(self.client_dict[cli][key]) > 1]))
                        clientlist.remove(cli) 
                if len(clientlist) == 0: LoadMoreHistory = False
    
        if self.debug: print "finished read in %0.1fs." % (time()-t0)

if __name__ == "__main__":
    ###  example execution::
    rec = mediaRecommend()

    clientlist = ['clientID...']
    rec.build_client_model(clientlist=clientlist,max_chunks=24,max_vids=10000)

    #domainlist = ['vimeo','youtube']
    #rec.build_domain_model(domainlist=domainlist,max_chunks=12,max_vids=15000)


    #pdb.set_trace()

