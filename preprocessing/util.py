# coding: utf-8
import json

def read_timelines(specified_mood, gender):
    raw_data_path = '../data/raw_dataset'
    intermediary_path = '../data/intermediary_data'
    if gender == 'male' or gender == 'female':
        gender_path = '%s/gender_score_%s.txt' % (intermediary_path, gender)
        gender_uids = set()
        with open(gender_path) as f:
            for line in f:
                gender_uids.add(int(line.strip().split()[0]))

    timeline_path = '%s/vader_rst.txt' % (raw_data_path)
    timelines = []
    with open(timeline_path) as f:
        for line in f:
            timeline = json.loads(line.strip())
            if timeline['mood'] != specified_mood:
                continue
            if (gender == 'male' or gender == 'female') and timeline['uid'] not in gender_uids:
                continue
            timelines.append(timeline)
    return timelines