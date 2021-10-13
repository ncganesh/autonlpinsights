
<h3 align="center">
    <p>Auto NLP Insights</p>
</h3>

---------------------------

AutoNLPInsights  extracts the Named Entities,Sentiments,Summary,KeyPhrases,Topics
from the (Url/Plain Text/PDF Files ) and helps to visualize them (EDA) with one line of code.

######  DASH APP
![img.png](img.png)


## Installation

    git clone https://github.com/ncganesh/autonlpinsights
    pip install -r requirements.txt

## Visualization

```python
from autonlpinsights import nlpinsights
# Any URL or Plain Text
data = 'https://www.cnbc.com/2021/08/06/doximity-social-network-for-doctors-full-of-antivax-disinformation.html'
nlpinsight = nlpinsights(data)

nlpinsight.cleanedtext

# WORDCLOUDS
nlpinsight.visualize_wordclouds()

# NGRAMS
nlpinsight.visualize_ngrams(ngram_value = 2,top_n=5)

# NAMED ENTITY TREE MAP
nlpinsight.visualize_namedentities()

# For Vizualizing Raw text with Named Entities  (Include spacy_fig = True)
nlpinsight.visualize_namedentities(spacy_fig = True) 

# SENTIMENTS (Pie with sentiment labels )
nlpinsight.visualize_sentiments()

# SENTIMENT TABLE (Sentences sorted with sentiment score along with labels)

nlpinsight.get_sentiment_df()

# SUMMARY(Top 5 sentences using Abstarctive Summarization)

nlpinsight.get_summary_table()

# Gensim Topic Modelling 

nlpinsight.visualize_topics(num_topics=3)
```
```


Note:
This is still in initial phase of Developement and will be adding more features soon
