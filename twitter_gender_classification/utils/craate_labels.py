

import sys, os

FEMALE = 0.0
MALE = 1.0
gen_class_encode = {'f':FEMALE, 'm':MALE}

def create_gender_labels(timelines_dir,out_path):
    with open(out_path,'w') as of:
        for t in os.listdir(timelines_dir):

            l = t.split('_')[2]
            of.write('{}\t{}\n'.format(t,l))

if __name__ == '__main__':

    create_gender_labels(sys.argv[1],sys.argv[2])

    lables = dict([l.split('\t') for l in  map(lambda x: x.strip(),open(sys.argv[2]).readlines())])
    for l in lables:
        print l, ' >>> ', lables[l]
