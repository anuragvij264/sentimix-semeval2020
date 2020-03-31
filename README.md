# sentimix-semeval2020

### Sentimix (SemEval 2020) submissions 

Problem Statement: Given Hindi-English codemixed tweets with their tags. Predict the sentiment of the tweets.


 ### Datasets used
 1. SemEval sentimix dataset 
 2. CFILT IITB Hi-En dataset 
 3. LOIT (Thanks to Praneeth for this)



### Feature Extraction methods employed:
  1. Plain word2vec and glove embeddings 
  2. Transliterated Hindi words to English and using fasttext to fetch embeddings.
  3. Deepmoji features extracted (using penultimate layer)
  4. Fasttext embeddigns trained on LOIT 
  5. TFIDF (bigrams and trigrams)
  6. Emoji scoring (count based)
  
### Models employed : 
1. Random Forest ( with TFIDFs as features) 
2. LSTM (with and without attention) 
3. BiLSTM (with and with attention)
4. CNN ( without attention) (* gave us best individual results) 
5. Ensembling 1-2,1-2-3, and 1-4
  
Code and notebooks for all the models and results and scores would be updated soon.
  
 
