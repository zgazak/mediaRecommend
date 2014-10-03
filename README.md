# mediaRecommender

A python object to create video recommendations based on "content" and "quality".  Pushes the recommendation model to tornado database "MediaRecommend" to enable querying the model.

The program works as follows (see build_model function to follow):

	1) load max_vids videos per each client for up to max_chunks events::times files

	2) calculate "quality" parameters:

		a) Focus "F": "clickiness" of a user... focus = 1 means user watched video straight through.  Focus = 0 means a user clicked around as much as we can measure.  
			- basic idea: if we get "progress events" of 0-5,5-15,15-30 and the timestamps of each show 5s, 10s, and 15s, that user watched straight through and has a focus score of 1. 
			- Instead if we see "0-1,9-10,6-8,120-121,122-123" and the timestamp shows an elapse of 5s, this is a focus score of 0

		b) Interest "I":  Tplay/Tvideo... watching full video = 1, half = 0.5, watching the video twice = 2, etc...

		c) Popularity "Pop": video's "plays" / overall plays for all considered videos.  

	3) calculate N "content topics" and measure each video's relationship to each topic.
		Method: construct a descriptive "document" for each video as follows:
			- add the uid key for a user to a video they watch X times, where X = percent watch /10 (a user that watches a video fully is added 10 times, half a video is 5, etc)
			- create a "bag of words" and "term frequency - inverse document frequency" (TF-IDF) representation of the "documents" for each video
			- use LDA (latent dirichlet allocation) to arrange the videos into N topics (N = number of videos / 50, with min = 5 and max = 40).  This gives each video a % of being a member of each "topic" ... so for 5 topics, a video will have a % string something like... [0.2, 0.1, 0.5, 0.05, 0.15]

	4) develop a relative probability of recommending every video to every other:
		Vr = video to recommend
		Vw = video being watched
		P( Vr | Vw) = F_Vr * I_Vr * Pop_Vr * sum(over topics t) of Vr_t*Vw_t

	 	In this way we weight "quality" and "topic" in the recommendation, driving users to videos of similar topic and high quality.  With this formulation we will occasionally recommend videos of different topics if they are close enough and very high quality.  

	5) push the top 10 recommended videos for each video to MediaRecommend tornado database, with relative weights of each recommendation.  The key for retrieving a recommendation for videoX is: APIkey_videoXhash, with videoXhash = urlbin.Url.parse(videoX_url).key and APIkey the client's API key.

Usage:
```python

# Initialize recommender object:
# debug will print progress and elapsed time information to stdout.
rec = mediaRecommend(debug=True)

# run recommender on client(s)
# and automatically update the Tornado database MediaRecommend
clientlist=['client_keyA','client_keyB']
rec.build_client_model(clientlist=clientlist)

## other parameters:
# max_chunks: maximum most recent events::time chunks to search through... default = 24 (12 hours).
# max_vids: maximum videos to consider per client in clientlist.  default=10,000.  This takes precedence over max_chunks... if max_vids is hit first for a client, older chunks are not added to that client's information.  

