import pandas as pd
import numpy as np
# We can use counter to count the objects
from collections import Counter



df = pd.read_csv("data/aspectGraphGeneration.csv")
print(df['SENTIMENT'])

df = df.loc[df['SENTIMENT'].isin(['Negative'])]
print(df['SENTIMENT'])
data_f = []
for item in df['ASPECTS']:
    data_f.append(item.replace('[', '').replace(']', '').replace("'",''))
data = ""
for row in data_f:
    data = data + row + ","

#print(data)
data1 = data.replace(',',' ')

from collections import Counter
from itertools import chain
import nltk
from nltk.corpus import stopwords
 
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

cnt = Counter()


for text in data1.split():
    if text not in stop_words:
        cnt[text] += 1
# See most common ten words
cnt.most_common(10)
print(cnt.most_common(20))

import pandas as pd
word_freq = pd.DataFrame(cnt.most_common(20),
                             columns=['words', 'count'])
word_freq.head()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 8))

# Plot horizontal bar graph
word_freq.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="red")
ax.set_title("Improvement Areas")
plt.show()