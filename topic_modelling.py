
def token(senten, word_tokenize):
    results = []
    for sentence in senten:
        results.append(word_tokenize(sentence))
    return results

def topic_model(reviews_lemmatized, gensim):
    np.random.seed(0)

    # initialize GSDMM
    gsdmm = MovieGroupProcess(K=15, alpha=0.1, beta=0.3, n_iters=15)

    # create dictionary of all words in all documents
    dictionary = gensim.corpora.Dictionary(reviews_lemmatized)

    # filter extreme cases out of dictionary
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    # create variable containing length of dictionary/vocab
    n_terms = len(dictionary)

    # fit GSDMM model
    model = gsdmm.fit(reviews_lemmatized, n_terms)
    doc_count = np.array(gsdmm.cluster_doc_count)

    # topics sorted by the number of document they are allocated to
    top_index = doc_count.argsort()[-15:][::-1]

    # show the top 20 words in term frequency for each cluster 
    top_words(gsdmm.cluster_word_distribution, top_index, 20)
    return top_index, gsdmm

def top_words(cluster_word_distribution, top_cluster, values):
    for cluster in top_cluster:
        sort_dicts =sorted(gsdmm.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        print("\nCluster %s : %s"%(cluster,sort_dicts))
    return

def create_topics_dataframe(data_text,  mgp, threshold, topic_dict, lemma_text):
    result = pd.DataFrame(columns=['Text', 'Topic', 'Lemma-text'])
    for i, text in enumerate(data_text):
        result.at[i, 'Text'] = text
        result.at[i, 'Lemma-text'] = lemma_text[i]
        prob = mgp.choose_best_label(reviews_lemmatized[i])
        if prob[1] >= threshold:
            result.at[i, 'Topic'] = topic_dict[prob[0]]
        else:
            result.at[i, 'Topic'] = 'Other'
    return result

def processing(data, gensim, malaya, word_tokenize, WordCloud ):
    df = data.iloc[:, 0]

    # change text abbreviations to original word
    df1 = df.str.replace(r'\bx\b', 'tidak')
    df1 = df1.str.replace(r'\btak\b', 'tidak')
    df1 = df1.str.replace(r'\borg\b', 'orang')
    df1 = df1.str.replace(r'\bdgn\b', 'dengan')
    df1 = df1.str.replace(r'\bmora\b', 'moratorium')
    df1 = df1.str.replace(r'\bni\b', 'ini')
    df1 = df1.str.replace(r'\btu\b', 'itu')

    # remove unwanted word
    df1 = df1.str.replace('\n', '')
    df1 = df1.str.replace(r'\bla\b', '')
    df1 = df1.str.replace(r'\bje\b', '') 

    # remove stopword
    stop_words = malaya.text.function.STOPWORDS
    df2 = df1.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    # dataframe change to list
    list_dat = df2.values.tolist()
    
    # tokenize word
    reviews_lemmatized = token(list_dat, word_tokenize)

    # GSDMM for the topic modeling
    top_index, gsdmm = topic_model(reviews_lemmatized, gensim)

    # give name to the cluster
    topic_dict = {}
    topic_names = ['type 1', 'type 2', 'type 3', 'type 4', 'type 5', 'type 6', 'type 7', 'type 8', 'type 9', 
                   'type 10', 'type 11', 'type 12', 'type 13', 'type 14', 'type 15']
    for i, topic_num in enumerate(top_index):
        topic_dict[topic_num]=topic_names[i]

    # create dataframe with topic
    result = create_topics_dataframe(data_text=df1, mgp=gsdmm, threshold=0.3, topic_dict=topic_dict, lemma_text=reviews_lemmatized)
    result['Lemma_text'] = result['Lemma-text'].apply(lambda row: ' '.join(row))
    result = result.drop('Lemma-text', axis=1)

    # create word clouds
    wc1 = create_WordCloud(data_q = result['Lemma_text'].loc[result.Topic == 'type 1'], title="Most used words in cluster 5", WordCloud)
    wc2 = create_WordCloud(data_q = result['Lemma_text'].loc[result.Topic == 'type 2'], title="Most used words in cluster 10", WordCloud)
    return wc1, wc2

def create_WordCloud(data_q, title=None, WordCloud):
    wordcloud = WordCloud(width = 500, height = 500,
                          collocations = False,
                          background_color ='white',
                          min_font_size = 15
                          ).generate(" ".join(data_q.values))
                      
    #plt.figure(figsize = (5, 5), facecolor = None) 
    #plt.imshow(wordcloud, interpolation='bilinear') 
    #plt.axis("off") 
    #plt.tight_layout(pad = 0) 
    #plt.title(title,fontsize=20)
    #plt.show()
    return wordcloud
