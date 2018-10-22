import sys, os, shutil, pickle

from utils.glove_preprocess import *
from utils.misc import *
from utils import ark_tagger
from sklearn.feature_selection import f_classif


def train_classifer(data_dir,labels_path, glove_embeds, anova_top_features, prob=True,calib=True):
    
    #print 'Classifying ...'

    (X, y, ids) = convert_to_X_y(data_dir, anova_top_features,labels_path)            
    
    feature_count = len(anova_top_features)
    glove_dim = len(glove_embeds[glove_embeds.keys()[10]])
    
    X = convert_to_glove_embeds(X, glove_embeds, anova_top_features)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    '''This is the header for classification performance data'''        
    clf_name= 'RF160'; clf =RandomForestClassifier(n_estimators=160)
    xmethod = 'glove_embeds' 
    if calib:clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid', cv=10)
    preds, probs,clf = run_cv(X, y, clf)
    clf.fit(X, y)
    metrics_ = compute_metrics(y, preds, probs)
    results = 'glove_dim: {}\tfeature_count: {}\tfeature_type: {}\tclf_type: {}\n{}'.format(glove_dim, feature_count, xmethod.ljust(10), clf_name, metrics_[0])
    print results    
    return clf

if __name__ == '__main__':

    ''' Srep 1: preprocess the timelines for GLOVE '''
    # 'tweet_texts_path' must contain a file for each timeline. IN each file, there is line for the text of the tweet.
    tweet_texts_path = sys.argv[1]
    glove_pre_processed_path = '_train_{}_glove_pre_processed'.format(tweet_texts_path.strip('/').split('/')[-1])
    if not os.path.exists(glove_pre_processed_path) or not all_fnames_match(tweet_texts_path, glove_pre_processed_path):
        os.mkdir(glove_pre_processed_path)
        glove_pre_process(tweet_texts_path, glove_pre_processed_path)

    ''' Srep 2: do pose tagging '''
    # Download ark-tweet-nlp from https://code.google.com/archive/p/ark-tweet-nlp/downloads
    tokenozed_dir = '_train_{}_tokenozed_dir'.format(tweet_texts_path.strip('/').split('/')[-1])
    ark_tokenizer_path = sys.argv[2] # this must be path to "runTagger.sh" that comes with ark-tweet-nlp
    if not os.path.exists(tokenozed_dir) or not all_fnames_match(tweet_texts_path,tokenozed_dir):
        os.mkdir(tokenozed_dir)
        ark_tagger.tagger(ark_tokenizer_path,glove_pre_processed_path,tokenozed_dir)

    ''' Srep 3: TFDIF weighting '''
    tfidf_out_dir = '_train_{}_tfidf_out_dir'.format(tweet_texts_path.strip('/').split('/')[-1])
    if os.path.exists(tfidf_out_dir):
        shutil.rmtree(tfidf_out_dir)
    os.mkdir(tfidf_out_dir)
    tfidf_transformer,high_df_features = tfidf_weighting(tokenozed_dir, tfidf_out_dir,threshold=20)

    ''' Srep 4: Train classifier '''
    # path to glove.twitter.27B.200d.txt
    # can be downloaded from https://nlp.stanford.edu/projects/glove/
    glove_path = sys.argv[3]

    # We load glove embeddings to make sure we apply anova on top of features with a glove embedding
    glove_embeds = load_glove(glove_path, high_df_features)
    features = glove_embeds.keys()

    labels_path=sys.argv[4]
    X,y,_ = convert_to_X_y(tfidf_out_dir, features,labels_path)
    top_features_cnt = 100
    anova_top_features = extract_best_features(X, y, f_classif, top_features_cnt, features)
    anova_top_features.reverse()
    anova_top_features = [str(f) for f in anova_top_features]

    glove_embeds =  {k:glove_embeds[k] for k in anova_top_features}
    #glove_embeds = load_glove(glove_path, anova_top_features)
    clf = train_classifer(tfidf_out_dir,labels_path, glove_embeds, anova_top_features)

    clf_path = sys.argv[5]
    with open(clf_path,'w') as of:
        pickle.dump((clf,tfidf_transformer,anova_top_features,high_df_features,glove_embeds),of)
            
    shutil.rmtree(tfidf_out_dir)