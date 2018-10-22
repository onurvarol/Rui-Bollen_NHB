# coding: utf-8
"""
This script computes the peak z-scores of each timeline in emotional period
This script has 2 parameters:
The first one is the specified mood. 1 for positive, -1 for negative
The second one is the gender. You can set it to "all", "male" or "female".
examples:
python diverge_test.py 1 all && python diverge_test.py -1 all
"""

import sys
import numpy as np
import util

def timeline_peak_zscore(timelines, specified_mood, gender):
    # the results store in the intermediary folder
    intermediary_path = '../data/intermediary_data'
    foutpath = '%s/user_peak_zscores_mood=%d_gender=%s.txt' % (intermediary_path, specified_mood, gender)
    with open(foutpath, 'w') as fout:
        for timeline in timelines:
            venting_stamp = timeline['stamp']
            # times and valences store the timestamps and valences of tweets in the timeline
            times = [tweet['stamp'] - venting_stamp for tweet in timeline['tweets'] if abs(tweet['stamp'] - venting_stamp) < 6 * 3600]
            valences = [tweet['mood'] for tweet in timeline['tweets'] if abs(tweet['stamp'] - venting_stamp) < 6 * 3600]

            if len(valences) > 1:
                mean, std = np.mean(valences), np.std(valences)
                if std > 0:
                    zscores = [(val - mean) / std for val in valences]
                    if specified_mood == 1:
                        start, end = -38, 53 # the CUSUM boundaries for positive emotion
                    else:
                        start, end = -63, 9 # the CUSUM boundaries for negative emotion
                    # cut the zscore list based on the boundary
                    boundary_zscores = [zscores[i] for i in range(len(zscores)) if start * 60 <= times[i] <= end * 60]
                    if boundary_zscores:
                        # pick max or min zscore according to the specified mood
                        if specified_mood == 1:
                            peak_val = max(boundary_zscores)
                        else:
                            peak_val = min(boundary_zscores)
                        fout.write('%d\t%f\n' % (timeline['uid'], peak_val))

if __name__ == '__main__':
    specified_mood = int(sys.argv[1])
    gender = sys.argv[2]

    # read timelines
    timelines = util.read_timelines(specified_mood, gender)

    # calculate the peak zscores for each timeline
    timeline_peak_zscore(timelines, specified_mood, gender)
