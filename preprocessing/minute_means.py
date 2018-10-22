# coding: utf-8

"""
This script computes the mean emotion score in each window. 1 minute window moves from -6 hours to +6 hours.
This script has 2 parameters:
The first one is the specified mood. 1 for positive, -1 for negative
The second one is the gender. You can set it to "all", "male" or "female".
examples:
python minute_means.py 1 all && python minute_means.py 1 male && python minute_means.py 1 female
python minute_means.py -1 all && python minute_means.py -1 male && python minute_means.py -1 female
"""

import sys
import numpy as np
import util

def slide_means(timelines, specified_mood, gender):
    """
    slide window by 1 minute and compute the mean emotion score in that window
    :param timelines: user timelines
    """
    window_size = 60 # window_size is 60 seconds / 1 minute
    step = window_size # step is equal with the size of window
    start, end = -6 * 3600, 6 * 3600 - window_size + 1 # the window slides from -6 hours to +6 hours

    # the results store in the intermediary folder
    intermediary_path = '../data/intermediary_data'
    foutpath = '%s/mean_mood=%d_window=%d_gender=%s.txt' % (intermediary_path,
                                                             specified_mood, window_size, gender)
    with open(foutpath, 'w') as fout:
        fout.write('start\tmean\n')
        # the window slides minute by minute
        for start_t in range(start, end, step):
            end_t = start_t + window_size
            scores = []
            for timeline in timelines:
                if timeline['mood'] == specified_mood:
                    for tweet in timeline['tweets']:
                        # if the tweet post in the 1 minute window, we collect it's valence
                        if start_t <= tweet['stamp'] - timeline['stamp'] < end_t:
                            scores.append(tweet['mood'])
            mean = np.mean(scores)
            fout.write('%d\t%f\n' % (start_t, mean))

if __name__ == '__main__':
    specified_mood = int(sys.argv[1])
    gender = sys.argv[2]

    # read timelines
    timelines = util.read_timelines(specified_mood, gender)

    # slide the window and calculate the average valence in each window
    slide_means(timelines, specified_mood, gender)


