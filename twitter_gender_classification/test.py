import sys, os, shutil, pickle

from utils.glove_preprocess import *
from utils.misc import *
from utils import ark_tagger

def apply_classifier(data_dir,glove_embeds,anova_top_features,clf,prob = True):
        
    (X, _, ids) = convert_to_X_y(data_dir, anova_top_features,None)
    X= convert_to_glove_embeds(X, glove_embeds, anova_top_features)
    if prob:
        preds = np.array(map(lambda p: p[clf.classes_.tolist().index(FEMALE)], clf.predict_proba(X)))
    else:
        preds = np.array(clf.predict(etrash_data[xmethod]))
    return preds,ids

if __name__ == '__main__':

    clf_path = sys.argv[1]
    with open(clf_path,) as _if:
        clf,tfidf_transformer,anova_top_features,high_df_features,glove_embeds = pickle.load(_if)

    ''' Srep 1: preprocess the timelines for GLOVE '''
    # 'tweet_texts_path' must contain a file for each timeline. IN each file, there is line for the text of the tweet.
    tweet_texts_path = sys.argv[2]
    glove_pre_processed_path = '_test_{}_glove_pre_processed'.format(tweet_texts_path.strip('/').split('/')[-1])
    if not os.path.exists(glove_pre_processed_path) or not all_fnames_match(tweet_texts_path, glove_pre_processed_path):
        os.mkdir(glove_pre_processed_path)
        glove_pre_process(tweet_texts_path, glove_pre_processed_path)

    ''' Srep 2: do pose tagging '''
    # Download ark-tweet-nlp from https://code.google.com/archive/p/ark-tweet-nlp/downloads
    tokenozed_dir = '_test_{}_tokenozed_dir'.format(tweet_texts_path.strip('/').split('/')[-1])
    ark_tokenizer_path = sys.argv[3] # this must be path to "runTagger.sh" that comes with ark-tweet-nlp
    if not os.path.exists(tokenozed_dir) or not all_fnames_match(tweet_texts_path,tokenozed_dir):
        os.mkdir(tokenozed_dir)
        ark_tagger.tagger(ark_tokenizer_path,glove_pre_processed_path,tokenozed_dir)

    ''' Srep 3: TFDIF weighting '''
    tfidf_out_dir = '_test_{}_tfidf_out_dir'.format(tweet_texts_path.strip('/').split('/')[-1])
    if os.path.exists(tfidf_out_dir):
        shutil.rmtree(tfidf_out_dir)
    os.mkdir(tfidf_out_dir)
    tf_idf_weighting_test(tokenozed_dir,tfidf_out_dir,tfidf_transformer,high_df_features)


    ''' Srep 4: Apply classifier '''
    # path to glove.twitter.27B.200d.txt
    # can be downloaded from https://nlp.stanford.edu/projects/glove/

    print "Applying the classifier ..."
    preds,ids = apply_classifier(tfidf_out_dir,glove_embeds,anova_top_features,clf)
    print "Finished classification"
    preds_save_path = sys.argv[4]
    with open(preds_save_path,'w') as of:
        for i,id in enumerate(ids):
            of.write('{}\t{}\n'.format(id,preds[i]))

    shutil.rmtree(tfidf_out_dir)    
