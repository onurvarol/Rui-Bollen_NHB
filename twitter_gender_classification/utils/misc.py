import sys, os, shutil
import codecs
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

FEMALE = 0.0
MALE = 1.0
gen_class_encode = {'f':FEMALE, 'm':MALE}
gen_class_decode = {FEMALE:'F', MALE:'M'}

def token_counter(in_path, start_date,end_date):
    words_dic = {}
    words_count = 0
    tokens = [l.split('\t')[0] for l in codecs.open(in_path, 'r', encoding='utf8').readlines()]
    tweet_count = 0
    i = 0
    while i < len(tokens):
        if tokens[i] == '\n':
            i+=1 
            continue
        if (i == 0 or tokens[i-1] == '\n'):
            within_date = True
            if start_date and end_date:
                date = datetime.strptime(tokens[i].strip(), '%Y-%m-%d %H:%M:%S')
                i+=1
                if not (date>=start_date and date<=end_date):
                    within_date = False
            if within_date:
                tweet_count += 1
        if not within_date:
            i+= 1
            continue
        if i < len(tokens) - 2 and tokens[i] == '<' and tokens[i + 2] == '>':
            w = '<' + tokens[i + 1].lower() + '>'
            i += 3
        else:
            w = tokens[i].lower()
            i += 1

        for c in map(unicode,[226, 128, 153, 10, 39]):w = w.replace( c, '')
        words_dic.update({w:words_dic.get(w, 0) + 1})
        words_count += 1
            
    return words_count, words_dic,tweet_count       


def tfidf_weighting(train_path, tfidf_out_dir, **kwargs):
    
    '''
    Filtering can take two different values: 'freq' and 'df'
    Timelines with less than 10 tweets are eliminated
    '''
    
    start_date=kwargs.setdefault('start_date',None) 
    end_date=kwargs.setdefault('end_date',None)
    filtering = kwargs.get('filtering','df')
    threshold = kwargs.get('threshold',30)
        
    print 'tfidf_weighting ...'
    
    all_features = {}
    collection_freq = {}
    timelines_word_counts = {}
    timlines_names = os.listdir(train_path)
    for idx,t in enumerate(timlines_names):
        in_path = os.path.join(train_path, t)
        words_count, words_dic, tweet_count = token_counter(in_path, None,None)
        timelines_word_counts.update({t:words_dic})
        for w in words_dic:
            all_features.update({w:all_features.get(w, 0) + 1})
            collection_freq.update({w:collection_freq.get(w, 0) + words_dic.get(w) })
        if ((idx+1) % 10 == 0) or (idx+1 == len(timlines_names)):print "Processed {}/{}".format(idx+1,len(timlines_names))
    
    print "Num all_features: = {}".format(len(all_features))
    print "Filtering low freq features ..."
    if filtering == 'df':
        high_df_features = sorted([f for f in all_features if all_features[f] > (threshold - 1)])
    elif filtering == 'freq':
        high_df_features = sorted([f for f in all_features if collection_freq[f] > (threshold - 1)])
    
    if kwargs.setdefault('features_store_path',None) is not None:
        with codecs.open(kwargs.setdefault('features_store_path',None), 'w', encoding='utf8') as out:
            for w in high_df_features:
                out.write(w);out.write('\n')
    
    print 'len(high_df_features): {}'.format(len(high_df_features)) 
    
    print "FITTING TFIDF TRANSFOMRATION ..."   
    y = timelines_word_counts.keys()[:]
    X = np.array([[timelines_word_counts[t].get(f, 0) for f in high_df_features] for t in y])
    tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
    tfidf_transformer.fit(X)
    
    
    
    for i,t in enumerate(y):
        x = tfidf_transformer.transform(X[i].reshape((1, X.shape[1]))).toarray().flatten()
        with codecs.open(os.path.join(tfidf_out_dir, t), 'w', encoding='utf8') as out:
            out.write('\n'.join([high_df_features[j] + '\t' + str(x[j])  for j in range(x.shape[0])]))
        if ((1+1) % 100 == 0) or (i+1 == len(y)):print '{} train timelines converted'.format(i+1)

    return tfidf_transformer,high_df_features


def tf_idf_weighting_test(test_dir,tfidf_out_dir,tfidf_transformer,features,**kwargs):
    print "Running tf_idf_weighting_test ..."
    start_date=kwargs.setdefault('start_date',None) 
    end_date=kwargs.setdefault('end_date',None)
    timlines_names = os.listdir(test_dir)
    for idx,t in enumerate(timlines_names):
        if t.endswith('~'):continue
        in_path = os.path.join(test_dir, t)
        words_count, words_dic, tweet_count = token_counter(in_path, start_date,end_date)
        #if tweet_count <30:continue
        x = np.array([words_dic.get(f, 0) for f in features])
        x = tfidf_transformer.transform(x.reshape((1, x.shape[0]))).toarray().flatten()
        with codecs.open(os.path.join(tfidf_out_dir, t), 'w', encoding='utf8') as out:
            out.write('\n'.join([features[j] + '\t' + str(x[j])  for j in range(x.shape[0])]))
        if ((idx+1) % 10 == 0) or (idx+1 == len(timlines_names)):print "Test timelines processed: {}/{}".format(idx+1,len(timlines_names))
    print "Finished tf_idf_weighting_test"

def convert_to_X_y(data_dir, feature_list,labels_path):
    '''
    This functions converts the dataset to X and y fomrat and filters the features which do not have a glove space vector
    In the returned X, the feature values are in the same order as in the supported 'feature_set' array
    ''' 
    def feature_value_of_timeline(timeline_path, feature_set):
        featue_val_map = {}
        with codecs.open(timeline_path, 'r', encoding='utf8') as lines:
            for line in lines:
                if len(line.strip().split('\t')) > 1:
                    [f, w] = line.strip().split('\t')
                    if not f in feature_set:
                        continue
                    featue_val_map.update({f:float(w)})
        return featue_val_map

    timelines = [os.path.join(data_dir, t) for t in os.listdir(data_dir) if not t.endswith('~')]
    X = np.zeros((len(timelines),len(feature_list)))
    feature_set = set(feature_list)
    for idx,t_path in enumerate(timelines):
        featue_val_map = feature_value_of_timeline(t_path, feature_set)
        timeline = [featue_val_map.get(f, 0) for f in feature_list]
        X[idx]=timeline
    
    ids = np.array([os.path.basename(t_path) for t_path in timelines ])
    if labels_path is None:
        y = np.zeros((len(timelines),))
    else:
        lables = dict([l.split('\t') for l in  map(lambda x: x.strip(),open(labels_path).readlines())])
        y = np.array([ gen_class_encode[lables[t]] for t in os.listdir(data_dir) if not t.endswith('~')])

    return X, y,ids

def run_cv(X, y, clf, num_cv_folds=10):
    from sklearn.cross_validation import KFold, cross_val_predict
    kfolds = KFold(len(y), n_folds=num_cv_folds)
    preds = [None] * len(y)
    probs = [None] * len(y)
    for (train, test) in kfolds:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        
        p = clf.fit(X_train, y_train).predict_proba(X_test)[:, clf.classes_.tolist().index(0.0)]
        for i in range(len(test)):
            probs[test[i]] = p[i]
        # preds
        p = clf.predict(X_test)
        for i in range(len(test)):
            preds[test[i]] = p[i]
            
    return preds, probs,clf

def convert_to_glove_embeds(X, glove_embeds, feature_list,norm=None):
    '''
    Converts given X space to X in glove space by adding up weighted sum of GLOVE vectors for each word in X
    It assumes that the supported glove space vecs in the argument only include glove space vecs for the supported
    array of best features.
    '''
    #print "Converting to GLOVE space"
    
    glove_vecs = np.array([glove_embeds[f] for f in feature_list])
    if norm is None:
        X_glove = np.array([np.sum(x.reshape((len(x), 1)) * glove_vecs, axis=0) for x in X])
    elif norm == 'word_count':
        X_glove = np.array([np.sum(x.reshape((len(x), 1)) * glove_vecs, axis=0)/len(x) for x in X])
    elif norm == 'tfidf_sum':  
        X_glove = np.array([np.sum(x.reshape((len(x), 1)) * glove_vecs, axis=0)/np.sum(x) for x in X])
        
    #print "Converting to GLOVE space DONE"
    return X_glove

def load_glove(glove_path, feature_list):
    print'Loading glove ...'
    '''
    Loads GLOVE vectors.
    Only keeps those features that are found in features_path text file.
    '''
    glove_embeds = {}
    glove_feature_count = 0
    
    feature_set = set(feature_list)
    with codecs.open(glove_path, 'r', encoding='utf8') as lines:
            for line in lines:
                f = line.split(' ')[0]
                if f in feature_set:
                    glove_embeds.update({f:np.array(line.split()[1:], dtype=np.float64)})
                glove_feature_count += 1

    for f in [u'', u'\n']:
        if f  in glove_embeds:
            glove_embeds.pop(f)
    
    print'finished loading glove'
    return glove_embeds

def extract_best_features(X, y, scor_func, number_of_features, features, alphabetic=False):
    '''''
    sort features based on their score and then
    get the k best and sort them based based on their index in feature list which is 
    alphabetic (the outer application of sorted())
    
    returned features are in ascending order, that is the last one is the best one 
    '''''
    #print 'Extract best features using ANOVE ... '
    from sklearn.feature_selection import SelectKBest
    k_ = number_of_features
    selector = SelectKBest(scor_func, k=k_).fit(X, y)
    best_features_alhphabetic_based = [features[i] for i in sorted(selector.scores_.argsort()[-k_:])]
    best_features_rank_based = [features[i] for i in selector.scores_.argsort()[-k_:]]
    
    #print 'Extract best features using ANOVE DONE'
    if alphabetic:
        return best_features_alhphabetic_based
    return best_features_rank_based

def compute_metrics(y, predictions, probablities):
    '''''
    Return accuracy precision recall roc 
    '''''
    from sklearn import metrics
    accuracy = metrics.accuracy_score(y, predictions)
    precision = (metrics.precision_score(y, predictions, average='binary', pos_label=0.0), metrics.precision_score(y, predictions, average='binary', pos_label=1.0))
    recall = (metrics.recall_score(y, predictions, average='binary', pos_label=0.0), metrics.recall_score(y, predictions, average='binary', pos_label=1.0))
    avg_prec = metrics.average_precision_score(1-y,probablities)
    
    roc = metrics.roc_auc_score(1-y, probablities)
    prec = tuple(map(lambda x: round(x, 3), precision))
    recall = tuple(map(lambda x: round(x, 3), recall))
    
    #results = 'Accuracy: {}  Precision:<{}, {}>  Recall: <{}, {}>  ROC_AUC: {}'.format(round(accuracy, 3),prec[0],prec[1] ,recall[0],recall[1] , round(roc, 3))
    results = 'accuracy: {}\taverage_precision:{}\tprecision 0: {}\tprecision 1: {}\trecall 0:{}\trecall 1:{}\troc: {}'\
                .format(round(accuracy, 3),round(avg_prec,3),prec[0],prec[1] ,recall[0],recall[1] , round(roc, 3)) 
    return results, accuracy,avg_prec, precision, recall, roc


def all_fnames_match(src,dst):

    src_names = set( [f for f in os.listdir(src) if not '~' in f])
    dst_names = set( [f for f in os.listdir(dst) if not '~' in f])

    return src_names == dst_names
