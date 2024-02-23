import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import download
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import json
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

data = pd.read_csv("data/final_merged_data.csv")
#print(data.head())
#print(data.info())

#Cleaning Data Sets
data['REVIEW_COMMENTS'] = data['REVIEW_COMMENTS'].map(lambda x: re.sub(r'[^a-zA-Z ] ', ' ', str(x)))
print(len(data['REVIEW_COMMENTS']))
data = data.drop_duplicates(subset=['REVIEW_COMMENTS'], keep='first')
print(len(data['REVIEW_COMMENTS']))
#Drop empty rows on Review
data = data.dropna(subset=['REVIEW_COMMENTS'])
print(len(data['REVIEW_COMMENTS']))
#find stop words, along with review realted stopwords; ex. stop = stopwords.words('english') + ['tsp', 'tbsp', 'finely','extra', 'chopped' ]
#----------Stop words to generate Negative word cloud
#stop = stopwords.words('english') + ['good','great','like','mind blowing','superb','satisfied','high quality','working properly','working','properly']

#----------Stop words to generate Positive word cloud
stop = stopwords.words('english')


#filtering dataframe with either positive or negative sentiment
sentiment = ['Positive']
updatedData = data.loc[data['OVERALL_SENTIMENT'].isin(sentiment)] 

#--------Word frequency count-------
wrdFreq = updatedData['REVIEW_COMMENTS'].str.lower().str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(wrdFreq)
word_dist = nltk.FreqDist(words)
print (word_dist)
print("\n")
rslt = pd.DataFrame(word_dist.most_common(100),columns=['Word', 'Frequency'])
print(rslt)
print("\n")

def remove_encoding_word(word):
    word = str(word)
    word = word.encode('ASCII', 'ignore').decode('ASCII')
    return word

def remove_encoding_text(text):
    text = str(text)
    text = ' '.join(remove_encoding_word(word) for word in text.split() if word not in stop)
    return text

#lemmatize the text
updatedData['REVIEW_COMMENTS'] = updatedData['REVIEW_COMMENTS'].apply(remove_encoding_text)
text = ' '.join(words for words in updatedData['REVIEW_COMMENTS'])
#print(len(text))

lemma = WordNetLemmatizer().lemmatize

#fit into the tf-idf model
def tokenize(document):
    tokens = [lemma(w) for w in document.split() if len(w) > 3 and w.isalpha()]
    return tokens

vectorizer = TfidfVectorizer(tokenizer = tokenize, ngram_range = ((2,2)), stop_words = stop, strip_accents = 'unicode')

tdm = vectorizer.fit_transform(updatedData['REVIEW_COMMENTS'])
#print(vectorizer.vocabulary_.items())

#finally lets create the wordcloud. 
tfidf_weights = [(word, tdm.getcol(idx).sum()) for word, idx in vectorizer.vocabulary_.items()]
print(tfidf_weights[0:10])

# Choose color maps from website: https://matplotlib.org/stable/users/explain/colors/colormaps.html
# Colormap for Positive = "Greens"
# Colormap for Negative = "Reds"
#collocations=False
w = WordCloud(width=1500, height=1200, mode='RGBA', background_color='white', max_words=2000, colormap="Greens").fit_words(dict(tfidf_weights))
plt.figure(figsize=(20,15))
plt.imshow(w)
plt.axis('off')
plt.savefig('wrdcloud/PositiveCloud.png')
