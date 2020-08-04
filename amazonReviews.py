# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
    import string
    import re
    from autocorrect import Speller
    import nltk
    from nltk.corpus import stopwords
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.tokenize import sent_tokenize, word_tokenize
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from nltk.stem import PorterStemmer 
    from knn import KNN
    from textblob import TextBlob
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD

#%%
    
  
    
    test_raw_data = []
    train_raw_data = []
    test_process_data = []
    train_process_data = []
    train_labels = [] 
    
    spell = Speller(lang='en')
    stop_words = nltk.corpus.stopwords.words('english')
    stemmer = PorterStemmer()
    
        
    #useful for removing punctuation
    spaces = ""
    for i in range(32):
        spaces = spaces + " ";


    #preprocessing test data
    with open ("test_master.dat") as file_test:
    
        for line in file_test:
            test_raw_data.append(line)
            
        for line in test_raw_data:
        
           #takes off punctuation 
           line = line.translate(str.maketrans(string.punctuation, spaces, '0123456789'))
           line = spell(line) #fix spelling 
    
           
           words = word_tokenize(line) #makes a line into a list
           polarity_words = []
            #process out words without any polarity
           for i, word in enumerate(words):
               sentiment = TextBlob(word).sentiment
               if(sentiment[0]> 0.3):
                   polarity_words.append(word)
               elif(sentiment[0]< -0.3):
                   polarity_words.append(word)
                                     
            #stemming each word
           for i, word in enumerate(polarity_words):
               polarity_words[i]=stemmer.stem(word)
               polarity_words[i]= word.lower()         
           line = ' '.join(polarity_words)
           
              
           test_process_data.append(line)
        print("done processing test data")
   
    
      
     
     #preprocessing train data
    with open("train_master.dat") as file_train:
        for line in file_train:
            train_raw_data.append(line) 
            
        for line in train_raw_data:  
           label = line.split()[0]       
           train_labels.append(label)
    
           #takes off punctuation
           line = line.translate(str.maketrans(string.punctuation, spaces, '0123456789'))        
           line = spell(line) #fix spelling
    
           
           words = word_tokenize(line) #makes a line into a list
           polarity_words = []
            #process out words without any polarity 
           for i, word in enumerate(words):
               sentiment = TextBlob(word).sentiment
               if(sentiment[0]> 0.3):
                   polarity_words.append(word)
               elif(sentiment[0]< -0.3):
                   polarity_words.append(word)
                   
           
            #stemming each word 
           for i, word in enumerate(polarity_words):
               polarity_words[i]=stemmer.stem(word) 
               polarity_words[i]= word.lower()
           line = ' '.join(polarity_words)  
              
           train_process_data.append(line)
           
        print("done processing train data")
        print()

    
    
 #%%   
      
 
    
    K= 137
#    X_train, X_test, y_train, y_test = train_test_split(train_process_data,train_labels, test_size = 0.2, random_state = 42)
    
    tfv = TfidfVectorizer(stop_words = 'english', min_df = 2)
    train_tfidf = tfv.fit_transform(train_process_data).toarray()
    test_tfidf = tfv.transform(test_process_data).toarray()
    
    tfv_features = tfv.get_feature_names()

    
    def predict(test_tfidf):
        predictions = [predict_each(test_tfidf[counter], counter) for counter in range(len(test_tfidf))]     
        return predictions
    
    
    def predict_each(i, counter):
        #compute distances
        i = i.reshape(1,-1)
        distances = [cosine_similarity(i, train_tfidf)]

        #get k nearest neighbors
        nearest = np.argsort(distances)[0][0][::-1][:K]
        nearest = nearest.flatten()
        
        predection_labels = [train_labels[i] for i in nearest]

        #get most common class label
        predection_labels = [int(i) for i in predection_labels]
        sum_all = sum(predection_labels)
        if(sum_all >0):
            most_common = '+1'
        else:
            most_common = '-1'

        counter+= 1
 
        print(str(counter)+" "+ most_common)
        return most_common
    
    distances = []
    predictions = predict(test_tfidf)
    
#    print(accuracy_score(y_test, predictions))

    
    #print predictions into a file 
    outfile = open("predictions.txt","w+")
    for i in predictions:
        outfile.write("%s\n" % i)
    
    outfile.close()
    
    
  
    

  
    
    

    
    
    
    
    
    
    
 
      
    
  
           
           
           
           
       
 




