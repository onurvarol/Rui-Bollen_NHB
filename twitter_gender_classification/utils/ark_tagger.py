
import sys, os
import subprocess

def tagger(tagger_script,timelines_dir,outdir):

    '''Ark POS Tagger needs Java runtime'''    
    print 'Running ark POS Tagger for each timeline ...'
    fnames = os.listdir(timelines_dir)
    for i,t in enumerate(fnames):
        print 'Processing {}/{}'.format(i+1,len(fnames))
        tpath = os.path.join(timelines_dir,t)
        outpath = os.path.join(outdir,t)
        with open(outpath,'w') as of:
            subprocess.call([tagger_script,"--output-format", "conll", tpath],shell=False,stdout=of)
    print 'Finished ark POS tagging.'
