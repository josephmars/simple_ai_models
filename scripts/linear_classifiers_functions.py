"""Functions for linear classifiers
"""



import re 
from os import makedirs, path


import nltk
import joblib
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


from scripts.linear_classifiers_functions import Tokenizer, Embedder


class Tokenizer():
  """
    Tokenizer for linear classifiers
  """

  def __init__(self):
    
    """Constructor method.


    """
    self.TOKEN_DEFINITION = ""
    self.VOCAB = []
    self.VOCAB_SIZE = 0
    self.LEMATIZE = False
    self.LEMATIZER = WordNetLemmatizer()


  def __call__(self,X, 
          to_lower = True):

    return tuple(self.transform(X,to_lower=to_lower))

  def fit(self,
          X, 
          to_lower = True, 
          min_token_aparison = 5,
          token_definition = r"(\.|!|\?|[a-z0-9]+)",
          lemmatize = True,
          stopwords = [])->None:

    """Trains a tokenizer

        :param [str|list] X: text to train with.
        :param boolean to_lower: pass the text to lower case
        :param int min_token_aparison: mininum number of aparison to consider a
          token in the vocabulary
        :param regexp token_definition: regexp to define a token 

    """
    self.lematize = lemmatize
    # type checking
    if type(X)==str:
      X = [X]
    elif type(X) == list:
      pass
    else:
      raise TypeError
    

    # pass to lowercase if needed
    if to_lower:
      tmp = X
      X = []

      for i in tmp:
        X.append(i.lower())

    # 
    vocabulary = {}
    
    #
    for x_i in X:
      tokens = re.findall(token_definition,x_i)
      for token in tokens:
        if(lemmatize):
          token = self.LEMATIZER.lemmatize(token)

        if(token not in stopwords):
          vocabulary[token] = vocabulary.get(token,0) + 1    
    
    # apply  filters
    
    ##  min tokens
    tmp_vocabulary = {}
    for key in vocabulary:
      if vocabulary[key]>min_token_aparison:
        tmp_vocabulary[key] = vocabulary[key]
    vocabulary = tmp_vocabulary

    #
    self.TOKEN_DEFINITION = token_definition
    self.VOCAB = [token for token in vocabulary]
    self.VOCAB = ['START'] + self.VOCAB + ['FINISH']  
    self.VOCAB_SIZE = len(self.VOCAB)
    return vocabulary

  def transform(self,
                X,
                to_lower=True
                )-> list:
    """Trains a tokenizer

        :param [str|list] X: text to train with.
        :param boolean to_lower: pass the text to lower case     
    """
    # type checking
    if type(X)==str:
      X = [X]
    elif type(X) == list:
      pass
    else:
      raise TypeError
    # pass to lowercase if needed
    if to_lower:
      tmp = X
      X = []
      for i in tmp:
        X.append(i.lower())
    
    # search

    final_words = []
    for x_i in X:
      words = []
      words.append("START")
      tokens = re.findall(self.TOKEN_DEFINITION,x_i)
      for token in tokens:
        if(self.LEMATIZE):
          token = self.LEMATIZER.lemmatize(token)
        if token in self.VOCAB:
          words.append(token)

      words.append("FINISH")
      final_words.append(words)
    return final_words

  def fit_transform(self,
          X, 
          to_lower = True, 
          min_token_aparison = 5,
          token_definition = r"(\.|!|\$|\?|[a-z0-9]+)")->None:

    """Trains a tokenizer

        :param [str|list] X: text to train with.
        :param boolean to_lower: pass the text to lower case
        :param int min_token_aparison: mininum number of aparison to consider a
          token in the vocabulary
        :param regexp token_definition: regexp to define a token 

    """
    # type checking
    if type(X)==str:
      X = [X]
    elif type(X) == list:
      pass
    else:
      raise TypeError
    # pass to lowercase if needed
    if to_lower:
      tmp = X
      X = []
      for i in tmp:
        X.append(i.lower())

    self.fit(X,to_lower=to_lower,
             min_token_aparison=min_token_aparison,
             token_definition=token_definition)
    
    raise NotImplemented

    return self
  
  def save(self,filename:str)->None:
    """saves the current tokenizer to disk.
        :param str filename: path to save the tokenizer


    """
    outfile = open(filename,'wb')
    pickle.dump(self,outfile)
    outfile.close()

    return None


  def load(filename):
    """loads a tokenizer from a path
        :param str filename: path to load the tokenizer

    """
    infile = open(filename,'rb')
    tokenizer_tmp = pickle.load(infile)
    infile.close()
    return tokenizer_tmp


class Embedder():

  def __init__(self,vocab:list):
    """ Bag of word embedding
      :param list vocab: vocabulary used

    """
    self.POSIBLE_MODES = ['ohe','tf','tfidf']
    self.vocab = vocab
    self.mode = ''
    self.idfs = None

  def fit(self,tokenized_x, mode = 'ohe') -> None:
    """Trains a bag of words embedding from a a list of tokenizeds sequences
        :param list[list] tokenized_x: tokenized list to train with
        :param str mode:  bag of word method to use

    """
    if(mode in self.POSIBLE_MODES):
      self.mode = mode
      if(mode == 'tfidf'):
        self.idfs = self.calc_idf(tokenized_x)
    else:
      raise NotImplementedError
    return None

  def transform(self,tokenized_x) -> np.array:
    """Applies the embedding to a list of tokenized sequences
        :param list[list] tokenized_x: list of tokenized sequences to embbed
        :return np.array: list of sequences vectors with the specified embeddeding method 

    """

    if self.mode == '':
      print("need to fit first")
      return None

    elif self.mode == 'ohe':
      return self.count_vectorized(tokenized_x)
    elif self.mode == 'tf':
      return self.count_vectorized(tokenized_x,binary=False)
    elif self.mode == 'tfidf':
      return self.count_vectorized_tfidf(tokenized_x)


  def count_vectorized(self,tokenized_X,binary=True) -> np.array:
    """Calculates frecuencies from a list of tokenized sequences
        :param list[list] tokenized_X: list of tokenized sequences
        :param boolean binary:  {'true': one hot encoding,'false':term frecuency }
        :return np.array: word vectors  
    """
    vocab = self.vocab
    keys = [key for key in vocab]
    results = []
    
    for x in tokenized_X:
      sequence_vector = [0]*len(vocab)

      if(not x == []):
          
        for token in x: 
          if(token in vocab):
            sequence_vector[keys.index(token)] += 1
          
      if binary:
        for i in range(len(sequence_vector)):
          if sequence_vector[i] != 0:
            sequence_vector[i]/=sequence_vector[i]
    
      results.append(sequence_vector)
      
    return results

  def calc_idf(self,tokenized_x) -> np.array:
    """Calcs inverse document frecuency given a list of tokenized sequences
        :param tokenized_X: list of tokenized sequences to train with
        :return np.array: list of idfs for the vocabulary
    """
    
    vocab = self.vocab
    idfs = np.zeros((len(vocab))) 
    number_of_documents = len(tokenized_x)
    count_matrix = np.array(
        self.count_vectorized(
            tokenized_x
          )
        )
    for i in range(len(count_matrix[0,:])):
      idfs[i] = number_of_documents/(1+np.sum(count_matrix[:,i])) # POSIBLE NAN
    idfs = np.log(idfs)

    return idfs

  def calc_tf(self,tokenized_x) -> np.array:
    vocab = self.vocab
    count_matrix = np.array(
        self.count_vectorized(
            tokenized_x,
            binary=False
        )
    )
    tfs = np.zeros(np.shape(count_matrix))
    for i in range(len(count_matrix[:,0])):
      for j in range(len(count_matrix[0,:])):
        tfs[i,j] = count_matrix[i,j]/np.sum(count_matrix[i,:])
    return tfs

  def count_vectorized_tfidf(self,tokenized_x):

    vocab=self.vocab
    tfs = self.calc_tf(tokenized_x)
    embedded = []

    for num_sequence in range(np.shape(tfs)[0]):
      embedded.append(tfs[num_sequence,:] * self.idfs)

    return np.array(embedded)


  def save(self,filename:str)->None:
    """
    """
    outfile = open(filename,'wb')
    pickle.dump(self,outfile)
    outfile.close()

    return None


  def load(filename):
    """
    """
    infile = open(filename,'rb')
    tokenizer_tmp = pickle.load(infile)
    infile.close()
    return tokenizer_tmp


def make_predictions(embedded_data: list,classifiers: dict)->dict:

    final_dict = {}
    for key in classifiers:
        final_dict[key] = classifiers[key].predict(embedded_data)

    return final_dict


def apply_word_count_encoding(tokenized_X,vocab,binary=True):

  keys = [key for key in vocab]
  results = []

  # run by all the sequence
  for x in tokenized_X:
    sequence_vector = [0]*len(vocab)
    print("reading line ",x)
    #run by the tokens from the sequence
    for token in x:
      print("searching token",token)
      if(token in vocab):
        print("token hallado")
        sequence_vector[keys.index(token)] += 1
        print(" current word vector ",sequence_vector," for token: ",token)
    if binary:
      for i in range(len(sequence_vector)):
        if sequence_vector[i] != 0:
          sequence_vector[i]/= sequence_vector[i]
  
    print("adding ",sequence_vector)
    results.append(sequence_vector)
    
  #sequence_vector[keys.index(token)] = 1
  return results


def train_model(model_name: str, embedding_method = 'tfidf', training_data = pd.DataFrame(), TRAINING_INPUT_COLUMN = "text", TRAINING_LABEL_COLUMN = "label"):
  nltk.download('stopwords')
  nltk.download('wordnet')
  nltk.download('omw-1.4')
  
  df_original = training_data
  categories = list(np.unique(df_original[TRAINING_LABEL_COLUMN]))

  # ETL
  df = df_original.copy()
  tokenizer = Tokenizer()
  tokenizer.fit(list(df[TRAINING_INPUT_COLUMN]),
                min_token_aparison=5,
                stopwords= list(nltk.corpus.stopwords.words('english'))
  )

  embedders = {}
  embedder = Embedder(tokenizer.VOCAB)

  df["tokenized"] = tokenizer.transform(list(df[TRAINING_INPUT_COLUMN]))

  embedders[embedding_method] = embedder
  embedders[embedding_method].fit(list(df["tokenized"]),mode=embedding_method)
  df[embedding_method] = list(embedders[embedding_method].transform(list(df["tokenized"])))

  if model_name == 'KNN':
    classifier  = KNeighborsClassifier(
        n_neighbors=5,
        metric='euclidean'
    )
  elif model_name == 'RFC':
    classifier = RandomForestClassifier(
    random_state=0)

  classifier.fit(
        X= list(df[embedding_method]),
        y= list(df[TRAINING_LABEL_COLUMN])
  )
    
  return [tokenizer, embedder, classifier]