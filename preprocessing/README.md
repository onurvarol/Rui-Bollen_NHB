# NHB_AffectLabeling


Codes in this folder read raw files from `data/raw_dataset/` and produce intermediary files to 
`data/intermediary_data` folder.

The basic input file is "data/raw_dataset/vader_rst.zip" which contain all affect labeling timelines. 
All the three scripts `minute_means.py`, `CI.py` and `diverge_test.py` are based on this file.

If you unzip `data/raw_dataset/vader_rst.zip`, you will get `data/raw_dataset/vader_rst.txt`. In this file, each line is a JSON-format timeline. An example of a timeline is as follows:

```
{
 'mood': 1,
 'stamp': 1316107170,
 'uid': 27656192,
 'tweets': [tweet_1, tweet_2]
}
```

This timeline contains tweets posted by a user during 48 hours. In t0, user post a venting tweet such as "I feel bad". "stamp" is the timestamp of venting tweet's created time. "uid" means the ID of the user who post the timeline.
Tweets posted by the same user between [t0-24h, t0+24h] are collected. "tweets" is a list including all tweets. An example of a tweet is like:

```
{
 'id': 114054958990622720,
 'mood': 0.7482,
 'stamp': 1316027795,
 'week_minutes': 3736
}
```

"id" means the ID of the tweet. "stamp" is the created time of the tweet. 

"week_minutes", which is useful in null-model calculation, represents the posting minute in a week. "mood" means the valence calculated by VADER.

`minute_means.py` computes the average emotion score in each window. 
Please run this script by `python minute_means.py 1 all && python minute_means.py 1 male && 
python minute_means.py 1 female && python minute_means.py -1 all && python minute_means.py -1 male && 
python minute_means.py -1 female`. The corresponding output files are `data/intermediary_data/mean_mood=*_window=*_gender=*.txt`
 The two parameters of this script are mood and gender. For mood, 1 and -1 represent positive and negative
 emotion respectively. For gender, "all", "male" and "female" means using all timelines, timelines posted by male
 users and by female users respectively. There are two columns in the result file. First one is the 
 start minute of the window, sencond one is the average valence of tweets posted in the window.
 

`CI.py` computes vader and null-model confidence interval in each 10 minute window. Please run this script by
`python CI.py 1 all && python CI.py 1 male && python CI.py 1 female && 
python CI.py -1 all && python CI.py -1 male && python CI.py -1 female`.
The parameters are still mood and gender and the result files are `pos_all.txt`, `pos_male.txt`, 
`pos_female.txt`, `neg_all.txt`, `neg_male.txt` and `neg_female.txt` located in `data/intermediary_data` folder.
Each result file contains 8 columns. First one is the window represented by the center of a window. 
The second to the forth columns are the confidence interval of real timelines, which are 2.5p, 50p and 97.5p.
The fifth to the seventh are the confidence interval of null-model. The last column is the number of tweets
posted in the corresponding window.

`diverge_test.py` computes the peak z-scores of each timeline in emotional period. Please run this script by
`python diverge_test.py 1 all && python diverge_test.py -1 all`. The result files are 
`data/intermediary_data/user_peak_zscores_mood=*_gender=*.txt`. 
Each result file has two columns. The first one is the ID of the user who post the timeline, the second
one is the peak z-score.