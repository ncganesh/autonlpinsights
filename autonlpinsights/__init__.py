
# Data Processing
import string
from pathlib import Path
import validators
import pandas as pd
from newspaper import Article

# NLP
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim import corpora, models
#from gensim.summarization import keywords
import pyLDAvis.gensim_models
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser



# NLP Viz
import pyLDAvis.gensim_models

#pyLDAvis.enable_notebook()
from spacy import displacy

# Data Visualization
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud

nlp = spacy.load("en_core_web_sm")
lemma = WordNetLemmatizer()


class nlpinsights:
    """
    Exploratory Data Analysis Of Text Data
    """

    def __init__(self, data, column_name=False, pdf_file=False):
        # Checks if Input text is a URL and extracts text
        if validators.url(str(data)):
            self.data = self.extract_text_fromurl(data)
        elif column_name:
            # self.full_df = data
            self.data = self.extract_text_fromcolumn(data, column_name)
        elif pdf_file:
            self.data = self.extract_text_frompdf(data)
        else:
            self.data = data
        self.cleanedtext = self.cleantext()

    def cleantext(self):
        """
        Performs Basic Text Preprocessing
        (Remove Stopwords,Punctuation and lemmatize text)

        """
        tokens = word_tokenize(self.data)
        no_stop = " ".join([i for i in tokens if i not in set(stopwords.words('english')) and len(i) > 3])
        no_punc = ''.join(ch for ch in no_stop if ch not in set(string.punctuation))
        cleaned = " ".join(lemma.lemmatize(word) for word in no_punc.split())
        # cleaned = " ".join(word for word in no_punc.split())
        return cleaned

    def extract_text_fromcolumn(self, data, column_name):
        finaltext = ''
        data = data.sample(500)
        for text in data[column_name]:
            finaltext = finaltext + str(text)
        return finaltext
    """
        def extract_text_frompdf(self, pdf_filepath):
        #textractor = Textractor(sentences=True)
        sentences = textractor(pdf_filepath)
        finaltext = ''
        for sentence in sentences:
            finaltext = finaltext + str(sentence)
        return finaltext
    """


    def extract_text_fromurl(self, url):
        """
        Extract the text from url using newspaper package
        """
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        return text

    def get_namedentities(self):
        """
        Gets Named Entities using Spacy
        """
        doc = nlp(str(self.cleanedtext))
        named_entities = [(x.text, x.label_) for x in doc.ents]
        namedentdf = pd.DataFrame(named_entities, columns=['Text', 'Label'])
        return namedentdf

    def get_sentiments(self, text):
        """
        Gets Sentiment Score Dic using Vader
        """
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(str(text))
        return scores

    def get_sentiment_labels(self, text):
        """
          Gets Sentiment Labels
         :param text: Raw Text(each sentence).
        """
        analyzer = SentimentIntensityAnalyzer()
        sentiment_dict = analyzer.polarity_scores(str(text))
        # decide sentiment as positive, negative and neutral
        if sentiment_dict['compound'] >= 0.05:
            return "Positive"

        elif sentiment_dict['compound'] <= - 0.05:
            return "Negative"

        else:
            return "Neutral"

    def topic_model(self, num_topics):
        """
        LDA Topic Modelling using Gensim
        """
        sents = sent_tokenize(self.cleanedtext)
        doc_clean = [doc.split() for doc in sents]
        dictionary = corpora.Dictionary(doc_clean)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
        lda_model = models.ldamodel.LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=20)
        return lda_model, doc_term_matrix, dictionary

    def get_topics(self, num_topics=3):
        """
        Gets the Topic Dic with words along with weights
        """
        lda_model, doc_term_matrix, dictionary = self.topic_model(num_topics)
        return lda_model.show_topics()

    def get_keyphrases(self):
        """
        Gets Key Phrases using Gensim
        """
        keyphraselis = keywords(self.cleanedtext).split('\n')
        keyphrase_df = pd.DataFrame(keyphraselis, columns=['KeyPhrases'])
        return keyphrase_df

    def get_summary(self):
        """
        Gets Summarized Sentences using Sumy
        """
        parser = PlaintextParser.from_string(self.data, Tokenizer('english'))

        lsa_summarizer = LsaSummarizer()
        lsa_summary = lsa_summarizer(parser.document, 5)

        sentences = []
        # Printing the summary
        for sentence in lsa_summary:
            sentences.append(str(sentence).strip())
        summary_df = pd.DataFrame(sentences, columns=['Summary'])
        return summary_df

    def get_full_nlpinsights(self):
        """

        """
        nlpinsightsdic = {}

        namedentitiesdata = self.get_namedentities()
        nlpinsightsdic['cleaned_text'] = self.cleanedtext
        nlpinsightsdic['namedentities'] = namedentitiesdata.to_dict(orient='records')
        nlpinsightsdic['summary'] = self.get_summary()
        nlpinsightsdic['sentiments'] = self.get_sentiments(self.data)
        nlpinsightsdic['sentiments_df'] = self.get_sentiment_table().to_dict(orient='records')
        nlpinsightsdic['keyphrases'] = self.get_keyphrases()[:10]

        return nlpinsightsdic

    ######################################################################################
    ############################ *** VISUALIZATIONS *** ################################
    ####################################################################################

    def treemap_chart(self, tree_data):
        """
        Method to get treemap chart.
        :param tree_data: DataFrame
        """
        fig = px.treemap(tree_data, path=['Label', 'Text'])
        fig.update_layout(
            # margin=dict(t=50, l=25, r=25, b=25),
            font=dict(
                size=30
            )
        )
        return fig

    def pie_chart(self, sent_values):
        """
        Method to get Pie Chart
        :param sent_values: Series Data

        """
        fig = go.Figure(data=[go.Pie(labels=["negative", "neutral", "positive"], values=sent_values,
                                     marker=dict(colors=["red", "grey", "green"],
                                                 line=dict(color='#070707', width=1)))])
        fig.update_layout(
            title="Sentiment",
            font=dict(
                size=12,
            ),
        )
        return fig

    def bar_chart(self, values, labels):
        trace1 = go.Bar(y=labels, x=values, orientation='h', marker=dict(color='#009EEA'))
        data = [trace1]
        layout = go.Layout(barmode="group", xaxis=dict(title='Counts'),
                           yaxis=dict(autorange="reversed"), showlegend=False, font=dict(size=13), bargap=0.7,
                           )
        ngram_fig = go.Figure(data=data, layout=layout)
        return ngram_fig

    def wordcloud_plot(self):
        my_wordcloud = WordCloud(
            background_color='white',
            # height=275
        ).generate(self.cleanedtext)

        fig_wordcloud = px.imshow(my_wordcloud, template='ggplot2')
        fig_wordcloud.update_xaxes(visible=False)
        fig_wordcloud.update_yaxes(visible=False)
        return fig_wordcloud

    def visualize_ngrams(self, ngram_value, top_n):
        """
        Method to get Pie Chart
        :param ngram_value: Int (1 for Unigram,2 for Bigrams,3 for Trigrams)
        top_n: Int (Top 5 values)
        """
        tokens = word_tokenize(self.cleanedtext)
        bigrams_series = (pd.Series(nltk.ngrams(tokens, ngram_value)).value_counts())[:top_n]
        bigrams_labels = ["_".join(t) for t in bigrams_series.index]

        ngram_fig = self.bar_chart(list(bigrams_series.values), bigrams_labels)
        return ngram_fig

    def visualize_wordclouds(self):
        my_wordcloud = WordCloud(
            background_color='white',
            height=275
        ).generate(self.cleanedtext)
        return px.imshow(my_wordcloud, template='ggplot2')

    def visualize_namedentities(self, save_fig=False, spacy_fig=False):
        if save_fig:
            svg = displacy.render(doc, style="ent")
            output_path = Path("assets/namedentity_spacyplot.html")
            output_path.open("w", encoding="utf-8").write(svg)
        if spacy_fig:
            doc = nlp(self.cleanedtext)
            namedentity_fig = displacy.render(doc, style="ent")
            return namedentity_fig
        namedentdf = self.get_namedentities()
        namedentitytree_fig = self.treemap_chart(namedentdf)
        return namedentitytree_fig

    def visualize_sentiments(self):
        scores = self.get_sentiments(self.cleanedtext)
        del scores['compound']
        sent_values = list(scores.values())
        sentimentpie_fig = self.pie_chart(sent_values)
        return sentimentpie_fig

    def visualize_topics(self, num_topics=3, save_fig=False):
        lda_model, doc_term_matrix, dictionary = self.topic_model(num_topics)
        viz = pyLDAvis.gensim_models.prepare(lda_model, doc_term_matrix, dictionary)
        if save_fig:
            pyLDAvis.save_html(viz, 'assets/lda.html')
        return viz

    def get_sentiment_df(self):
        """

        """
        sents = sent_tokenize(self.data)
        sentiment_df = pd.DataFrame(sents, columns=['text'])
        sentiment_df['sentiment_compound_score'] = sentiment_df['text'].apply(
            lambda x: self.get_sentiments(x)['compound'])
        sentiment_df['sentiment_label'] = sentiment_df['text'].apply(lambda x: self.get_sentiment_labels(x))
        sentiment_df_sorted = sentiment_df.sort_values('sentiment_compound_score')
        return sentiment_df_sorted

    def get_summary_table(self):
        summary_df = self.get_summary()
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(summary_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[summary_df.Summary],
                       fill_color='lavender',
                       align='left'))
        ])

        # fig = ff.create_table(summary_df, height_constant=50)
        return fig

    def get_keyphrases_table(self):
        keyphrases_list = self.get_keyphrases()
        summary_df = pd.DataFrame(keyphrases_list, columns=['KeyPhrases'])
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(summary_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[summary_df.KeyPhrases],
                       fill_color='lavender',
                       align='left'))
        ])

        # fig = ff.create_table(summary_df, height_constant=50)
        return fig

