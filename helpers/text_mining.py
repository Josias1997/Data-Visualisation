############ Sentiment Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # visualization
import seaborn as sns
from datetime import datetime
import plotly.offline as py # visualization
import plotly.graph_objs as go # visualization
import plotly.tools as tls # visualization
import plotly.figure_factory as ff # visualization
# text manipulation
import  re
from nltk.corpus import stopwords

# Plot 3
# wordcloud - hashtags
from wordcloud import WordCloud

import networkx as nx
####### favorite and retweets by sentiment
import itertools
###### Support Vector Machine
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
######## Document Term Matrix
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
###### Classifier Random forest Algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import recall_score,precision_score,f1_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,GRU,LSTM,Embedding
from keras.optimizers import Adam
from keras.layers import SpatialDropout1D,Dropout,Bidirectional,Conv1D,GlobalMaxPooling1D,MaxPooling1D,Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
####### Network analysis of tweets
from sklearn.feature_extraction.text import CountVectorizer
###### Logistic regression 
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from .utils import generate_graph_img


def sentimental_analysis(df):

  # Select columns i will use for the analysis
  tweets = df[[ 'handle', 'text', 'is_retweet', 'original_author', 
                   'time', 'lang', 'retweet_count', 'favorite_count']]

  # Exploring the dataset    
  tweets.head()
  tweets.describe()
  tweets.info()

  ##### Data Manipulation
  # convert to date format and extract hour

  date_format = "%Y-%m-%dT%H:%M:%S" 
  tweets["time"]   = pd.to_datetime(tweets["time"],format = date_format)
  tweets["hour"]   = pd.DatetimeIndex(tweets["time"]).hour
  tweets["month"]  = pd.DatetimeIndex(tweets["time"]).month
  tweets["day"]    = pd.DatetimeIndex(tweets["time"]).day
  tweets["month_f"]  = tweets["month"].map({1:"JAN",2:"FEB",3:"MAR",
                                          4:"APR",5:"MAY",6:"JUN",
                                          7:"JUL",8:"AUG",9:"SEP"})

  # Language Used
  def label_language(df) :
      if df["lang"] == "en" :
          return "English"
      elif df["lang"] == "es" :
          return "French"
      else :
          return "Other"
  tweets["lang"] = tweets.apply(lambda tweets:label_language(tweets),axis = 1)

  # Create new tweets column colum tweets has same text than column text
  tweets["tweets"] = tweets["text"]


  stop_words = stopwords.words("english")

  # Function to remove special characters, punctions, stop words, digits, hyperlinks and case conversion
  def string_manipulation(df,column)  : 
      # extract hashtags
      df["hashtag"]  = df[column].str.findall(r'#.*?(?=\s|$)')
      # extract twitter account references
      df["accounts"] = df[column].str.findall(r'@.*?(?=\s|$)')
      
      # remove hashtags and accounts from tweets
      df[column] = df[column].str.replace(r'@.*?(?=\s|$)'," ")
      df[column] = df[column].str.replace(r'#.*?(?=\s|$)'," ")
      
      # convert to lower case
      df[column] = df[column].str.lower()
      # remove hyperlinks
      df[column] = df[column].apply(lambda x:re.split('https:\/\/.*',str(x))[0])
      # remove punctuations
      df[column] = df[column].str.replace('[^\w\s]'," ")
      # remove special characters
      df[column] = df[column].str.replace("\W"," ")
      # remove digits
      df[column] = df[column].str.replace("\d+"," ")
      # remove under scores
      df[column] = df[column].str.replace("_"," ")
      # remove stopwords
      df[column] = df[column].apply(lambda x: " ".join([i for i in x.split() 
                                                        if i not in (stop_words)]))
      return df

  tweets = string_manipulation(tweets,"text")

  # Trump tweets without retweets
  tweets_trump = (tweets[(tweets["handle"] == "realDonaldTrump") &
                           (tweets["is_retweet"] == False)].reset_index().drop(columns = ["index"],axis = 1))

  # trump tweets with retweets
  tweets_trump_retweets = (tweets[(tweets["handle"] == "realDonaldTrump") &
                                    (tweets["is_retweet"] == True)].reset_index().drop(columns = ["index"],axis = 1))

  tweets_trump.head(4).style.set_properties(**{}).set_caption("Trump tweets")

  # hillary tweets without retweets
  tweets_hillary  = (tweets[(tweets["handle"] == "HillaryClinton") &
                              (tweets["is_retweet"] == False)].reset_index()
                                .drop(columns = ["index"],axis = 1))

  # hillary tweets with retweets
  tweets_hillary_retweets  = (tweets[(tweets["handle"] == "HillaryClinton") &
                              (tweets["is_retweet"] == True)].reset_index()
                                .drop(columns = ["index"],axis = 1))

  tweets_hillary.head(4).style.set_properties(**{}).set_caption("Hillary tweets")



  #### Percentage of retweets

  plt.style.use('ggplot')

  plt.figure(figsize = (13,6))

  # Plot 1
  plt.subplot(121)
  tweets[tweets["handle"] ==
         "realDonaldTrump"]["is_retweet"].value_counts().plot.pie(autopct = "%1.0f%%",
                                                                  wedgeprops = {"linewidth" : 1,
                                                                                "edgecolor" : "k"},
                                                                  shadow = True,fontsize = 13,
                                                                  explode = [.1,0.09],
                                                                  startangle = 20,
                                                                  colors = ["blue","w"]
                                                                 )
  plt.ylabel("")
  plt.title("Percentage of retweets - Trump")

  # Plot 2
  plt.subplot(122)
  tweets[tweets["handle"] ==
         "HillaryClinton"]["is_retweet"].value_counts().plot.pie(autopct = "%1.0f%%",
                                                                  wedgeprops = {"linewidth" : 1,
                                                                                "edgecolor" : "k"},
                                                                  shadow = True,fontsize = 13,
                                                                  explode = [.09,0],
                                                                  startangle = 60,
                                                                  colors = ["red","w"]
                                                                 )
  plt.ylabel("")
  plt.title("Percentage of retweets - Hillary")
  retweets = generate_graph_img(plt)


  #### Languages used in tweet
  plt.figure(figsize = (12,7))

  # plot 1
  plt.subplot(121)
  ax = sns.countplot(y = tweets[tweets["handle"] == "realDonaldTrump"]["lang"] ,
                     linewidth = 1,edgecolor = "k"*3,
                     palette = "Reds_r")

  for i,j in enumerate(tweets[tweets["handle"] == 
                              "realDonaldTrump"]["lang"].value_counts().values) :
      ax.text(.7,i,j,fontsize = 15)

  plt.grid(True)
  plt.title("Languages used in tweets - trump")

  # Plot 2    
  plt.subplot(122)
  ax1 = sns.countplot(y = tweets[tweets["handle"] == "HillaryClinton"]["lang"] ,
                     linewidth = 1,edgecolor = "k"*3,
                      palette = "Blues_r")

  for i,j in enumerate(tweets[tweets["handle"] == 
                              "HillaryClinton"]["lang"].value_counts().values) :
      ax1.text(.7,i,j,fontsize = 15)

  plt.grid(True)
  plt.ylabel("")
  plt.title("Languages used in tweets - hillary")
  languages_used = generate_graph_img(plt)


  #######@ original authors of retweets
  plt.figure(figsize = (10,14))

  # Plot 1
  plt.subplot(211)
  authors = tweets_trump_retweets["original_author"].value_counts().reset_index()
  sns.barplot(y = authors["index"][:15] , 
              x = authors["original_author"][:15] ,
              linewidth = 1,edgecolor = "k",color = "#FF3300")
  plt.grid(True)
  plt.xlabel("count")
  plt.ylabel("original author")
  plt.title("original authors of retweets - Trump")

  # Plot 2
  plt.subplot(212)
  authors1 = tweets_hillary_retweets["original_author"].value_counts().reset_index()
  sns.barplot(y = authors1["index"][:15] , 
              x = authors1["original_author"][:15] ,
              linewidth = 1,edgecolor = "k",color ="#6666FF")
  plt.grid(True)
  plt.xlabel("count")
  plt.ylabel("original author")
  plt.title("original authors of retweets - Hillary")
  original_authors_retweets = generate_graph_img(plt)

  ####### tweets by month
  plt.figure(figsize = (12,8))
  sns.countplot(x = "month_f",hue = "handle",palette = ["#FF3300","#6666FF"],
                data = tweets.sort_values(by = "month",ascending = True),
               linewidth = 1,edgecolor = "k"*tweets_trump["month"].nunique())
  plt.grid(True)
  plt.title("tweets by month (2016)")
  tweets_by_month = generate_graph_img(plt)

  ####### Import positive an negative words dictionaries
  # PENSER A TRADUIRE TOUS SES MOTS EN FRANCAIS

  # positive words
  positive_words = pd.read_csv("helpers/datasets/positive_words.txt", header=None)
  # negative words
  negative_words = pd.read_csv("helpers/datasets/negative_words.txt", header=None)

  # Convert words to lists
  def convert_words_list(df) : 
      words = string_manipulation(df,0)
      words_list = words[words[0] != ""][0].tolist()
      return words_list

  positive_words_list = convert_words_list(positive_words)

  # Remove word trump from positive word list
  positive_words_list = [i for i in positive_words_list if i not in "trump"]
  negative_words_list = convert_words_list(negative_words)

  print ( "positive words : " ) # Afficher les 100 premiers mots positifs de la liste
  print (positive_words_list[:100])
  print ( "negative words : " )
  print (negative_words_list[:100]) # Afficher les 100 premiers mots négatifs de la liste

  # Scoring tweets
  # scoring tweets based on positive and negative words count.
  # score = positive_count - negative_count

  # function to score tweets based on positive and negative words present
  def scoring_tweets(data_frame,text_column) :
      #identifying +ve and -ve words in tweets
      data_frame["positive"] = data_frame[text_column].apply(lambda x:" ".join([i for i in x.split() 
                                                                                if i in (positive_words_list)]))
      data_frame["negative"] = data_frame[text_column].apply(lambda x:" ".join([i for i in x.split()
                                                                                if i in (negative_words_list)]))
      #scoring
      data_frame["positive_count"] = data_frame["positive"].str.split().str.len()
      data_frame["negative_count"] = data_frame["negative"].str.split().str.len()
      data_frame["score"]          = (data_frame["positive_count"] -
                                      data_frame["negative_count"])
      
      # Create a new feature sentiment :
      # Positive if score is positive , Negative if score is negative , Neutral if score is 0
      def labeling(data_frame) :
          if data_frame["score"] > 0  :
              return "positive"
          elif data_frame["score"] < 0  :
              return "negative"
          elif data_frame["score"] == 0 :
              return "neutral"
      data_frame["sentiment"] = data_frame.apply(lambda data_frame:labeling(data_frame),
                                                 axis = 1)
          
      return data_frame

  tweets = scoring_tweets(tweets,"text")
  tweets_trump = scoring_tweets(tweets_trump,"text")
  tweets_hillary = scoring_tweets(tweets_hillary,"text")

  tweets[["text","positive","negative","positive_count",
                "negative_count","score","sentiment"]].head()


  ##### Scores distribution
  score_dist = tweets[tweets["is_retweet"] ==
                      False].groupby("handle")["score"].value_counts().to_frame()
  score_dist.columns = ["count"]
  score_dist = score_dist.reset_index().sort_values(by = "score",ascending = False)

  trace = go.Bar(x = score_dist[score_dist["handle"] == "realDonaldTrump"]["score"],
                 y = score_dist[score_dist["handle"] == "realDonaldTrump"]["count"],
                 marker = dict(line = dict(width = 1,color = "black"),
                               color = "red"),name = "Donald Trump")

  trace1 = go.Bar(x = score_dist[score_dist["handle"] == "HillaryClinton"]["score"],
                  y = score_dist[score_dist["handle"] == "HillaryClinton"]["count"],
                  marker = dict(line = dict(width = 1,color = "black"),
                               color = "blue"),name = "Hillary Clinton")

  layout = go.Layout(dict(title = "Scores distribution",
                          plot_bgcolor  = "rgb(243,243,243)",
                          paper_bgcolor = "rgb(243,243,243)",
                          xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                       gridwidth = 2),
                          yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                       gridwidth = 2),
                          )
                    )
  fig = go.Figure(data = [trace,trace1], layout = layout)


  ######## Sentiment distribution of tweets
  sent_dist = pd.crosstab(tweets[tweets["is_retweet"] == 
                                 False]["sentiment"],
                          tweets[tweets["is_retweet"] ==
                                 False]["handle"]).apply(lambda r: r/r.sum()*100,axis = 0)

  sent_dist = sent_dist.reset_index()

  t1 = go.Bar(x = sent_dist["sentiment"],y = sent_dist["HillaryClinton"],
              name = "Hillary Clinton",
              marker = dict(line = dict(width = 1,color = "#000000"),color = "#6666FF"))

  t2 = go.Bar(x = sent_dist["sentiment"],y = sent_dist["realDonaldTrump"],
             name = "Donald Trump",
             marker = dict(line = dict(width = 1,color = "#000000"),color = "#FF3300"))

  layout = go.Layout(dict(title = "Sentiment distribution",
                          plot_bgcolor  = "rgb(243,243,243)",
                          paper_bgcolor = "rgb(243,243,243)",
                          xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                       gridwidth = 2,title = "sentiment"),
                          yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                       gridwidth = 2,title = "percentage"),
                          )
                    )
  fig = go.Figure(data = [t1,t2],layout = layout)


  ####### Correlation matrix
  df_corr = tweets[tweets["is_retweet"] == False][[ 'retweet_count', 'favorite_count' ,
                          'score', "sentiment","handle" ]]

  df_corr["neutral"]   = np.where(df_corr["score"] == 0,1,0)
  df_corr["negative"]  = np.where(df_corr["score"] < 0,1,0) 
  df_corr["positive"]  = np.where(df_corr["score"] > 0,1,0)

  cols = ['retweet_count','favorite_count','neutral','negative', 'positive']

  correlation_hillary  = df_corr[df_corr["handle"] == "HillaryClinton"][cols].corr()
  correlation_trump    = df_corr[df_corr["handle"] == "realDonaldTrump"][cols].corr()

  plt.figure(figsize = (12,4.5))
  plt.subplot(121)
  sns.heatmap(correlation_hillary,annot = True,cmap = "hot_r",
              linecolor = "grey",linewidths = 1)
  plt.title("Correlation matrix - Hillary")

  plt.subplot(122)
  sns.heatmap(correlation_trump,annot = True,cmap = "hot_r",
              linecolor = "grey",linewidths = 1)
  plt.title("Correlation matrix - Trump")
  correlation_matrix = generate_graph_img(plt)

  ####### Popular hashtags
  hashs_t = tweets_trump["tweets"].str.extractall(r'(\#\w+)')[0].value_counts().reset_index()
  hashs_t.columns = ["hash","count"]

  hashs_h = tweets_hillary["tweets"].str.extractall(r'(\#\w+)')[0].value_counts().reset_index()
  hashs_h.columns = ["hash","count"]

  # Plot 1
  plt.figure(figsize = (10,20))
  plt.subplot(211)
  ax = sns.barplot(x = "count" , y = "hash" ,
                   data = hashs_t[:25] , palette = "seismic",
                   linewidth = 1 , edgecolor = "k"* 25)
  plt.grid(True)
  for i,j in enumerate(hashs_t["count"][:25].values) :
      ax.text(3,i,j,fontsize = 10,color = "white")
  plt.title("Popular hashtags used by trump")

  # Plot 2
  plt.subplot(212)
  ax1 = sns.barplot(x = "count" , y = "hash" ,
                   data = hashs_h[:25] , palette = "seismic",
                   linewidth = 1 , edgecolor = "k"* 25)
  plt.grid(True)
  for i,j in enumerate(hashs_h["count"][:25].values) :
      ax1.text(.3,i,j,fontsize = 10,color = "white")
  plt.title("Popular hashtags used by hillary")
  popular_hashtags_used = generate_graph_img(plt)


  hsh_wrds_t = tweets_trump["tweets"].str.extractall(r'(\#\w+)')[0]
  hsh_wrds_h = tweets_hillary["tweets"].str.extractall(r'(\#\w+)')[0]

  def build_word_cloud(words,back_color,palette,title) :
      word_cloud = WordCloud(scale = 7,max_words = 1000,
                             max_font_size = 100,background_color = back_color,
                             random_state = 0,colormap = palette
                            ).generate(" ".join(words))
      plt.figure(figsize = (13,8))
      plt.imshow(word_cloud,interpolation = "bilinear")
      plt.axis("off")
      plt.title(title)
      return generate_graph_img(plt)


  hsh_wrds_trump = build_word_cloud(hsh_wrds_t,"black","rainbow","Hashtags - Trump")
  hsh_wrds_hillary = build_word_cloud(hsh_wrds_h,"black","rainbow","Hashtags - Hillary")

  ###### Popular twitter account references
  accounts_t = tweets_trump["tweets"].str.extractall(r'(\@\w+)')[0].value_counts().reset_index()
  accounts_t.columns = ["accounts","count"]

  accounts_h = tweets_hillary["tweets"].str.extractall(r'(\@\w+)')[0].value_counts().reset_index()
  accounts_h.columns = ["accounts","count"]

  plt.figure(figsize = (10,20))

  # Plot 1
  plt.subplot(211)
  ax = sns.barplot(x = "count" , y = "accounts" ,
                   data = accounts_t[:25] , palette = "seismic",
                   linewidth = 1 , edgecolor = "k"* 25)
  plt.grid(True)
  for i,j in enumerate(accounts_t["count"][:25].values) :
      ax.text(3,i,j,fontsize = 10,color = "white")
  plt.title("Popular twitter  account references by Trump")

  # Plot 2
  plt.subplot(212)
  ax1 = sns.barplot(x = "count" , y = "accounts" ,
                   data = accounts_h[:25] , palette = "seismic",
                   linewidth = 1 , edgecolor = "k"* 25)
  plt.grid(True)
  for i,j in enumerate(accounts_h["count"][:25].values) :
      ax1.text(3,i,j,fontsize = 10,color = "white")
  plt.title("Popular twitter  account references by Hillary")
  popular_twitter_account_references = generate_graph_img(plt)


  ##### wordcloud - accounts
  acc_wrds_t = tweets_trump["tweets"].str.extractall(r'(\@\w+)')[0]    
  acc_wrds_h = tweets_hillary["tweets"].str.extractall(r'(\@\w+)')[0]    

  acc_wrds_trump = build_word_cloud(acc_wrds_t,"black","rainbow","twitter account references - Trump")
  acc_wrds_hillary = build_word_cloud(acc_wrds_h,"black","rainbow","twitter account references - Hillary")


  ######## popular words in tweets
  pop_wrds_t = (tweets_trump["text"].apply(lambda x : 
                                           pd.value_counts(x.split(" ")))
              .sum(axis = 0).reset_index().sort_values(by = [0],ascending = False))
  pop_wrds_t.columns = ["word","count"]
  pop_wrds_t["word"] = pop_wrds_t["word"].str.upper()

  pop_wrds_d = (tweets_hillary["text"].apply(lambda x :
                                             pd.value_counts(x.split(" ")))
              .sum(axis = 0).reset_index().sort_values(by = [0],ascending = False))
  pop_wrds_d.columns = ["word","count"]
  pop_wrds_d["word"] = pop_wrds_d["word"].str.upper()

  plt.figure(figsize = (12,25))

  # Plot 1
  plt.subplot(211)
  ax = sns.barplot(x = "count",y = "word",data = pop_wrds_t[:30],
                  linewidth = 1 ,edgecolor = "k"*30,palette = "Reds")
  plt.title("popular words in tweets - Trump")
  plt.grid(True)
  for i,j in enumerate(pop_wrds_t["count"][:30].astype(int)) :
      ax.text(.8,i,j,fontsize = 9)

  # Plot 2    
  plt.subplot(212)
  ax1 = sns.barplot(x = "count",y = "word",data = pop_wrds_d[:30],
                  linewidth = 1 ,edgecolor = "k"*30,palette = "Blues")
  plt.title("popular words in tweets - Hillary")
  plt.grid(True)
  for i,j in enumerate(pop_wrds_d["count"][:30].astype(int)) :
      ax1.text(.8,i,j,fontsize = 9)

  # word cloud - popular words
  pop_wrds_trump = build_word_cloud(pop_wrds_t["word"],"black","Set1","popular words in tweets - Trump")
  pop_wrds_hillary = build_word_cloud(pop_wrds_d["word"],"black","Set1","popular words in tweets - Hillary")


  ########## Popular positive and negative words used by trump and Hillary
  def word_count(data_frame,column) :
      words = data_frame[column].str.split(expand = True)
      words = words.stack().reset_index()[0].value_counts().reset_index()
      words.columns = ["words","count"]
      words = words.sort_values(by = "count",ascending = False)
      words["words"] = words["words"].str.upper()
      return words
      
  pop_pos_words_t = word_count(tweets_trump,"positive")
  pop_neg_words_t = word_count(tweets_trump,"negative")

  pop_pos_words_d = word_count(tweets_hillary,"positive")
  pop_neg_words_d = word_count(tweets_hillary,"negative")

  ### Plot 1
  plt.figure(figsize = (12,22))
  plt.subplot(221)
  ax1 = sns.barplot(x = "count" , y = "words" ,
                   data = pop_pos_words_t[:20] , 
                   linewidth = 1 , edgecolor = "k"* 20)
  plt.grid(True)
  for i,j in enumerate(pop_pos_words_t["count"][:20].values) :
      ax1.text(8,i,j,fontsize = 10)
  plt.title("Popular positive words used by trump")

  ##### Plot 2
  plt.subplot(222)
  ax2 = sns.barplot(x = "count" , y = "words" ,
                   data = pop_neg_words_t[:20] , 
                   linewidth = 1 , edgecolor = "k"* 20)
  plt.grid(True)
  for i,j in enumerate(pop_neg_words_t["count"][:20].values) :
      ax2.text(8,i,j,fontsize = 10)
  plt.ylabel("")
  plt.title("Popular negative words used by trump")


  ###### Plot 3
  plt.subplot(223)
  ax3 = sns.barplot(x = "count" , y = "words" ,
                   data = pop_pos_words_d[:20] , 
                   linewidth = 1 , edgecolor = "k"* 20)
  plt.grid(True)
  for i,j in enumerate(pop_pos_words_d["count"][:20].values) :
      ax3.text(8,i,j,fontsize = 10)
  plt.title("Popular positive words used by hillary")

  ##### Plot 4
  plt.subplot(224)
  ax4 = sns.barplot(x = "count" , y = "words" ,
                   data = pop_neg_words_d[:20] , 
                   linewidth = 1 , edgecolor = "k"* 20)
  plt.grid(True)
  for i,j in enumerate(pop_neg_words_d["count"][:20].values) :
      ax4.text(8,i,j,fontsize = 10)
  plt.ylabel("")
  plt.title("Popular negative words used by hillary")

  plt.subplots_adjust(wspace = .3)
  popular_negative_words = generate_graph_img(plt)

  ######## Hashtag references by twitter accounts
  accounts = tweets["tweets"].str.extractall(r'(\@\w+)')[0].reset_index()[["level_0",0]]
  hash_tag = tweets["tweets"].str.extractall(r'(\#\w+)')[0].reset_index()[["level_0",0]]
  lf = hash_tag.merge(accounts,left_on = "level_0",right_on = "level_0",how = "left")[["0_x","0_y"]]
  rt = accounts.merge(hash_tag,left_on = "level_0",right_on = "level_0",how = "left")[["0_x","0_y"]]
  lf = lf.rename(columns = {"0_y" : "accs","0_x" : "hashs"})[["hashs","accs"]]
  rt = rt.rename(columns = {"0_x" : "accs","0_y" : "hashs"})[["hashs","accs"]]
  newdat = pd.concat([lf,rt],axis = 0)

  def connect_hash_acc(word,connect_type) :
      
      if connect_type == "hashtag_to_account" : 
          df = newdat[newdat["hashs"] == word]
          df = df[df["accs"].notnull()]
      elif connect_type == "account_to_hashtag" : 
          df = newdat[newdat["accs"] == word] 
          df = df[df["hashs"].notnull()]
          
      G  = nx.from_pandas_edgelist(df,"hashs","accs")
      plt.figure(figsize = (13,10))
      nx.draw_networkx(G,with_labels = True,font_size = 10,
                       font_color = "k",
                       font_family  = "DejaVu Sans",
                       node_shape  = "h",node_color = "b",
                       node_size = 1000,linewidths = 10,
                       edge_color = "grey",alpha = .6)
      

  connect_hash_acc("@realDonaldTrump","account_to_hashtag")
  connect_hash_acc("@FoxNews","account_to_hashtag")
  connect_hash_acc("#MakeAmericaGreatAgain","account_to_hashtag")
  connect_hash_acc("#RNCinCLE","hashtag_to_account")
  connect_hash_acc("#MAGA","hashtag_to_account")

  ######### Positive word references by Trump
  pw_t =  tweets_trump["positive"].str.split(expand = True).stack().reset_index()[0].str.upper()
  pw_d =  tweets_hillary["positive"].str.split(expand = True).stack().reset_index()[0].str.upper()
  pw_trump = build_word_cloud(pw_t,"black","cool","positive word references by trump")
  pw_hillary = build_word_cloud(pw_d,"black","cool","positive word references by hillary")

  ######## negative word references by trump
  nw_t =  tweets_trump["negative"].str.split(expand = True).stack().reset_index()[0].str.upper()
  nw_d =  tweets_hillary["negative"].str.split(expand = True).stack().reset_index()[0].str.upper()
  nw_trump = build_word_cloud(nw_t,"black","cool","negative word references by trump")
  nw_hillary = build_word_cloud(nw_d,"black","cool","negative word references by hillary")

  ######## Sentiment of tweets by hour of day
  st_hr_t = pd.crosstab(tweets_trump["hour"],tweets_trump["sentiment"])
  st_hr_t = st_hr_t.apply(lambda r:r/r.sum()*100,axis = 1)

  st_hr_d = pd.crosstab(tweets_hillary["hour"],tweets_hillary["sentiment"])
  st_hr_d = st_hr_d.apply(lambda r:r/r.sum()*100,axis = 1)

  # Plot 1
  st_hr_t.plot(kind = "bar",figsize = (14,7),color = ["r","b","g"],
                linewidth = 1,edgecolor = "w",stacked = True)
  plt.legend(loc = "best",prop = {"size" : 13})
  plt.title("Sentiment of tweets by hour of day - Trump")
  plt.xticks(rotation = 0)
  plt.ylabel("percentage")

  # Plot 2
  st_hr_d.plot(kind = "bar",figsize = (14,7),color = ["r","b","g"],
                linewidth = 1,edgecolor = "w",stacked = True)
  plt.legend(loc = "best",prop = {"size" : 13})
  plt.title("Sentiment of tweets by hour of day - hillary")
  plt.xticks(rotation = 0)
  plt.ylabel("percentage")
  sentiment_of_tweets = generate_graph_img(plt)


  lst =  ['negative', 'positive' ,'neutral']
  cs  =  ["r","g","b"]

  plt.figure(figsize = (13,13))

  for i,j,k in itertools.zip_longest(lst,range(len(lst)),cs) :
      
      plt.subplot(2,2,j+1)
      
      plt.scatter(x = tweets_trump[tweets_trump["sentiment"] == i]["favorite_count"],
                  y = tweets_trump[tweets_trump["sentiment"] == i]["retweet_count"],
                  label = "Trump",linewidth = .7,edgecolor = "w",s = 60,alpha = .7)
      
      plt.scatter(x = tweets_hillary[tweets_hillary["sentiment"] == i]["favorite_count"],
                  y = tweets_hillary[tweets_hillary["sentiment"] == i]["retweet_count"],
                  label = "Hillary",linewidth = .7,edgecolor = "w",s = 60,alpha = .7)
      
      plt.title(i + " - tweets")
      plt.legend(loc = "best",prop = {"size":12})
      plt.xlabel("favorite count")
      plt.ylabel("retweet count")


  ##### Average retweets and favorites by sentiment
  avg_fv_rts_t = tweets_trump.groupby("sentiment")[["retweet_count", "favorite_count"]].mean()
  avg_fv_rts_h = tweets_hillary.groupby("sentiment")[["retweet_count", "favorite_count"]].mean()

  # Plot 1
  avg_fv_rts_t.plot(kind = "bar",figsize = (12,6),linewidth = 1,edgecolor = "k")
  plt.xticks(rotation = 0)
  plt.ylabel("average")
  plt.title("Average retweets and favorites by sentiment - Trump")

  # Plot 2
  avg_fv_rts_h.plot(kind = "bar",figsize = (12,6),linewidth = 1,edgecolor = "k")
  plt.xticks(rotation = 0)
  plt.ylabel("average")
  plt.title("Average retweets and favorites by sentiment - Hillary")

  average_retweets = generate_graph_img(plt)



  def return_dtm(df,column) :
      
      documents  = df[column].tolist()
      vectorizer = CountVectorizer()
      vec = vectorizer.fit_transform(documents)
      dtm  = pd.DataFrame(vec.toarray(), columns = vectorizer.get_feature_names())
      dtm  = df[[column,"sentiment"]].merge(dtm, left_index = True, right_index = True, how = "left")
      dtm["sentiment"]  = dtm["sentiment"].map({"neutral" : 1,"positive" : 2, "negative" : 3})  
      
      return dtm

  dtm_trump = return_dtm(tweets_trump,"text")
  dtm_hillary = return_dtm(tweets_hillary, "text")
  # Rename column tetx_x by text
  dtm_hillary = dtm_hillary.rename(columns = {"text_x" : "text"})


  dtm_trump.head(3).style.set_properties(**{}).set_caption("DTM - Trump")
  dtm_hillary.head(3).style.set_properties(**{}).set_caption("DTM - Hillary")

  #############################################################################
  ################# Modeling and Machine learning #############################
  #############################################################################

  def split_data(dtm_df) :
      
      # Dependent variables and Independent variables
      predictors = [i for i in dtm_df.columns if i not in ["text"] + ["sentiment"]]
      target     = "sentiment"
      
      # Split the dataset into traing set and test set
      train, test = train_test_split(dtm_df, test_size = .25,
                                    stratify = dtm_df[["sentiment"]],
                                    random_state  = 123)
      # Define predictors and target on the training set and test set
      X_train = train[predictors]
      Y_train = train[target]
      X_test  = test[predictors]
      Y_test  = test[target]
      
      return X_train, Y_train, X_test, Y_test

  X_train_trp, Y_train_trp, X_test_trp, Y_test_trp = split_data(dtm_trump)
  X_train_hil, Y_train_hil, X_test_hil, Y_test_hil = split_data(dtm_hillary)

  # Plot 
  x      = [Y_train_trp, Y_test_trp, Y_train_hil, Y_test_hil]
  titles = ["train_data - trump","test_data - trump",
            "train_data - hillary","test_data - hillary"]

  plt.figure(figsize = (12,12))
  for i,j,k in itertools.zip_longest(x,range(len(x)),titles) :
      plt.subplot(2,2,j+1)
      counts = i.value_counts().reset_index()
      counts.columns = ["sentiment","count"]
      counts["sentiment"] = counts["sentiment"].map({1 : "neutral",2 : "positive" ,
                                                      3 : "negative" }) 
      plt.pie(x = counts["count"] ,labels = counts["sentiment"],autopct  = "%1.0f%%",
              wedgeprops = {"linewidth" : 1,"edgecolor" : "black"},
              colors = sns.color_palette("RdBu",7)) # color = muted, RdBu, prism, Blues_d
      plt.title(k)



  def classifier(X_train,Y_train,X_test,Y_test) :
      rfc = RandomForestClassifier(max_depth = 1000,max_features = 2000,
                                   n_estimators = 10,random_state = 123)
      rfc.fit(X_train, Y_train)
      predictions = rfc.predict(X_test)
      
      print ("accuracy_score  : ",accuracy_score(predictions, Y_test))
      print ("recall_score    : ",recall_score(predictions,Y_test,average = "macro"))
      print ("precision_score : ",precision_score(predictions, Y_test, average = "macro"))
      print ("f1_score        : ",f1_score(predictions,Y_test, average = "macro"))

      plt.figure(figsize = (8,6))
      sns.heatmap(confusion_matrix(predictions, Y_test),annot = True,
                  xticklabels= [ "neutral","positive","negative"],
                  yticklabels= [ "neutral","positive","negative"],
                  fmt = "d",linecolor = "w",linewidths = 2)
      plt.title("confusion matrix")
      return generate_graph_img(plt)

  # Classify trump tweets
  classifier_trump = classifier(X_train_trp, Y_train_trp, X_test_trp, Y_test_trp)

  # Classify hillary tweets
  classifier_hillary = classifier(X_train_hil, Y_train_hil, X_test_hil, Y_test_hil)



  def network_tweets(df,frequency,color,title) :
      # documents
      documents  = df[df["lang"] == "English"]["text"].tolist()
      vectorizer = CountVectorizer()
      vec        = vectorizer.fit_transform(documents)
      vec_t      = vectorizer.fit_transform(documents).transpose()
      
      # adjecency matrix for words
      adj_mat    = pd.DataFrame((vec_t * vec).toarray(),
                                columns = vectorizer.get_feature_names(),
                                index    = vectorizer.get_feature_names()
                               )
      # stacking combinations
      adj_mat_stack   = adj_mat.stack().reset_index()
      adj_mat_stack.columns = ["link_1","link_2","count"]
      
      # drop same word combinations
      adj_mat_stack   = adj_mat_stack[adj_mat_stack["link_1"] !=
                                      adj_mat_stack["link_2"]] 
      
      # subset dataframe with combination count greater than 25 times
      network_sub = adj_mat_stack[adj_mat_stack["count"] > frequency]
      
      # plot network
      H = nx.from_pandas_edgelist(network_sub,"link_1","link_2",["count"],
                                  create_using = nx.DiGraph())

      ax = plt.figure(figsize = (11,11))
      nx.draw(H,with_labels = True,alpha = .7,node_shape = "H",
              width = 1,node_color = color,
              font_weight = "bold",style = "solid", arrowsize = 15 ,
              font_color = "white",linewidths = 10,edge_color = "grey",
              node_size = 1300,pos = nx.kamada_kawai_layout(H))
      plt.title(title,color = "white")
      ax.set_facecolor("k")
      return generate_graph_img(plt)
      
  tweets_trump = network_tweets(tweets_trump,25,"#FF3300","Network analysis of tweet words - Trump")
  tweets_hillary = network_tweets(tweets_hillary,25,"#6666FF","Network analysis of tweet words - Hillary")

  ############ autres modeles ###############################################


  def log_reg(X_train,Y_train,X_test,Y_test) :
      
      logistic_regression = LogisticRegression(random_state = 0)
      logistic_regression.fit(X_train, Y_train)
      predictions = logistic_regression.predict(X_test)
      
      print ("accuracy_score  : ",accuracy_score(predictions,Y_test))
      print ("recall_score    : ",recall_score(predictions,Y_test,average = "macro"))
      print ("precision_score : ",precision_score(predictions,Y_test,average = "macro"))
      print ("f1_score        : ",f1_score(predictions,Y_test,average = "macro"))

      plt.figure(figsize = (8,6))
      sns.heatmap(confusion_matrix(predictions, Y_test),annot = True,
                  xticklabels= [ "neutral","positive","negative"],
                  yticklabels= [ "neutral","positive","negative"],
                  fmt = "d",linecolor = "w",linewidths = 2)
      plt.title("confusion matrix")
      return generate_graph_img(plt)

  log_reg_trump = log_reg(X_train_trp, Y_train_trp, X_test_trp, Y_test_trp)
  log_reg_hillary = log_reg(X_train_hil, Y_train_hil, X_test_hil, Y_test_hil)


  def svm(X_train,Y_train,X_test,Y_test) :
      
      svm = LinearSVC()
      svm.fit(X_train, Y_train)
      predictions = svm.predict(X_test)
      
      print ("accuracy_score  : ",accuracy_score(predictions, Y_test))
      print ("recall_score    : ",recall_score(predictions,Y_test,average = "macro"))
      print ("precision_score : ",precision_score(predictions,Y_test,average = "macro"))
      print ("f1_score        : ",f1_score(predictions,Y_test,average = "macro"))
      
      plt.figure(figsize = (8,6))
      sns.heatmap(confusion_matrix(predictions, Y_test),annot = True,
                  xticklabels= [ "neutral","positive","negative"],
                  yticklabels= [ "neutral","positive","negative"],
                  fmt = "d",linecolor = "w",linewidths = 2)
      plt.title("confusion matrix")
      
      print("Classification_report: ", classification_report(predictions, Y_test))
      return generate_graph_img(plt)

  svm_trump = svm(X_train_trp, Y_train_trp, X_test_trp, Y_test_trp)
  svm_hillary = svm(X_train_hil, Y_train_hil, X_test_hil, Y_test_hil)

  return {
    'retweets': f'data:image/png;base64,{retweets}',
    'languages_used': f'data:image/png;base64,{languages_used}',
    'original_authors_retweets': f'data:image/png;base64,{original_authors_retweets}',
    'tweets_by_month': f'data:image/png;base64,{tweets_by_month}',
    'correlation_matrix': f'data:image/png;base64,{correlation_matrix}',
    'popular_hashtags_used': f'data:image/png;base64,{popular_hashtags_used}',
    'hsh_wrds_trump': f'data:image/png;base64,{hsh_wrds_trump}',
    'hsh_wrds_hillary': f'data:image/png;base64,{hsh_wrds_hillary}',
    'popular_twitter_account_references': f'data:image/png;base64,{popular_twitter_account_references}',
    'acc_wrds_trump': f'data:image/png;base64,{acc_wrds_trump}',
    'acc_wrds_hillary': f'data:image/png;base64,{acc_wrds_hillary}',
    'pop_wrds_trump': f'data:image/png;base64,{pop_wrds_trump}',
    'pop_wrds_hillary': f'data:image/png;base64,{pop_wrds_hillary}',
    'popular_negative_words': f'data:image/png;base64,{popular_negative_words}',
    'pw_trump': f'data:image/png;base64,{pw_trump}',
    'pw_hillary': f'data:image/png;base64,{pw_hillary}',
    'nw_trump': f'data:image/png;base64,{nw_trump}',
    'nw_hillary': f'data:image/png;base64,{nw_hillary}',
    'sentiment_of_tweets': f'data:image/png;base64,{sentiment_of_tweets}',
    'average_retweets': f'data:image/png;base64,{average_retweets}',
    'classifier_trump': f'data:image/png;base64,{classifier_trump}',
    'classifier_hillary': f'data:image/png;base64,{classifier_hillary}',
    'tweets_trump': f'data:image/png;base64,{tweets_trump}',
    'tweets_hillary': f'data:image/png;base64,{tweets_hillary}',
    'log_reg_trump': f'data:image/png;base64,{log_reg_trump}',
    'log_reg_hillary': f'data:image/png;base64,{log_reg_hillary}',
    'svm_trump': f'data:image/png;base64,{svm_trump}',
    'svm_hillary': f'data:image/png;base64,{svm_hillary}',
  }









