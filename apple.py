import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


icon = Image.open('apple-xxl.png')
st.set_page_config(page_title="APPLE PRODUCT REVIEW ANALYSIS",
                   page_icon=icon,
                   layout="wide",
                   initial_sidebar_state="expanded",
                   )
st.markdown("<h1 style='text-align: center; color: #051937;background-color:white;border-radius:15px;'>APPLE PRODUCT REVIEW ANALYSIS AND PREDICTION</h1>",
            unsafe_allow_html=True)


def back_ground():
    st.markdown(f""" <style>.stApp {{
                        background-image: linear-gradient(to right top, #051937, #051937, #051937, #051937, #051937);;
                        background-size: cover}}
                     </style>""", unsafe_allow_html=True)


back_ground()


with st.sidebar:
    selected = option_menu(None, ["ANALYSIS", "N-GRAMS", "PREDICTION"],
                           icons=["bi bi-clipboard-data",
                                  "bi bi-body-text", "bi bi-magic"],
                           default_index=0,
                           orientation="vertical",
                           styles={"nav-link": {"font-size": "20px", "text-align": "centre", "margin-top": "20px",
                                                "--hover-color": "#266c81"},
                                   "icon": {"font-size": "20px"},
                                   "container": {"max-width": "6000px"},
                                   "nav-link-selected": {"background-color": "#266c81"}, })

df = pd.read_csv('APPLE_iPhone_SE.csv')
df_1 = df.copy()
df_1['isduplicate'] = df_1['Reviews'].duplicated()
df_2 = df_1[df_1['isduplicate'] == False]
df_2['Text_length'] = df_2['Reviews'].apply(len)


def clean_text(doc):
    comment_words = ""
    for i in doc:
        val = str(i)
        tokens = val.split(" ")
        for k in range(len(tokens)):
            tokens[k] = tokens[k].lower()
        comment_words = comment_words+" ".join(tokens)+""
        comment_words = ''.join(
            [i for i in comment_words if not i.isdigit()])
        comment_words = ''.join(
            [i for i in comment_words if i not in string.punctuation])

    return comment_words


df_2['Processed_txt'] = df_2['Reviews'].apply(lambda x: clean_text(x))


def get_wrods(doc):
    comment_words = ""
    for i in doc['Processed_txt']:
        val = str(i)
        tokens = val.split()
        for k in range(len(tokens)):
            tokens[k] = tokens[k].lower()
        comment_words = comment_words+" ".join(tokens)+" "
        comment_words = ''.join(
            [i for i in comment_words if not i.isdigit()])
    return comment_words


def plot_wordcloud(doc):
    from wordcloud import WordCloud, STOPWORDS
    stopwords = set(STOPWORDS)
    comment_words = get_wrods(doc)
    wordcloud = WordCloud(font_step=2, scale=3, width=2000, height=1000, background_color='white',
                          min_font_size=20, collocations=False, stopwords=stopwords).generate(comment_words)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud)
    ax.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig)


def get_top_n_words(corpus, n=2):
    vec = CountVectorizer(stop_words='english',
                          ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.fit_transform(corpus)
    sum_words = bag_of_words.toarray().sum(axis=0)
    word_list = vec.get_feature_names_out()
    doc = dict(zip(word_list, sum_words))
    return doc


def dict_to_dataframe(doc, n, m):
    common_words = get_top_n_words(doc, n)
    word_list = sorted(common_words.items(),
                       key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(word_list[:m], columns=['Words', 'Frequency'])
    return df


df_2['Review_cat'] = df_2['Ratings'].map(
    {1: 'Negative', 2: 'Negative', 3: 'Negative', 4: 'Positive', 5: 'Positive'})

df_class_pos = df_2[df_2['Review_cat'] == 'Positive']
df_class_neg = df_2[df_2['Review_cat'] == 'Negative']
under_sam_pos = df_class_pos.sample(df_class_neg.shape[0])
df_3 = pd.concat([under_sam_pos, df_class_neg], axis=0)
X = df_3['Processed_txt']
y = df_3['Review_cat']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


if selected == 'ANALYSIS':
    st.markdown("## :white[SAMPLE DATA]")
    st.table(df.head(10))

    st.markdown('## :white[DATA DESCRIPTION]')
    st.table(df_2.groupby('Ratings').describe().reset_index())
    ratings = df.groupby('Ratings').count().reset_index()
    fig, ax = plt.subplots()
    ax = sns.barplot(data=ratings, x='Ratings', y='Reviews')
    st.pyplot(fig)

    st.markdown('## :white[RATINGS WISE DISTRIBUTION]')
    g = sns.FacetGrid(df_2, col='Ratings')
    st.pyplot(g.map(plt.hist, 'Text_length'))

    st.markdown('## :white[WORD CLOUD FOR NEGATIVE REVIEW]')
    doc = df_2[(df_2['Ratings'] == 1) | (df_2['Ratings'] == 2)]
    plot_wordcloud(doc)
    st.markdown('## :white[IMPROVEMENTS NEEDED IN THE FOLLOWING AREAS]')
    st.markdown('#### :white[Customers having Issue with the battery life]')
    st.markdown('#### :white[the phone is getting heated fast]')
    st.markdown(
        '#### :white[some Customers are complaining about the charger,adapter]')
    st.markdown(
        '#### :white[some Customers are not statisfied with camera,speaker and display]')
    st.markdown(
        '#### :white[Customers are having issues with delivery partner like Flipkart Service]')

    st.markdown('## :white[WORD CLOUD FOR NEUTRAL REVIEW]')
    doc = df_2[df_2['Ratings'] == 3]
    plot_wordcloud(doc)

    st.markdown('## :white[WORD CLOUD FOR POSITIVE REVIEW]')
    doc = df_2[(df_2['Ratings'] == 4) | (df_2['Ratings'] == 5)].head(1000)
    plot_wordcloud(doc)
    st.markdown('## :white[AREAS OF STRENGTHS OF THE PRODUCT]')
    st.markdown('#### :white[phone is small and handy looking good]')
    st.markdown('#### :white[the charger is fast]')
    st.markdown('#### :white[camera is good]')
    st.markdown('#### :white[the display is compact and phone is handy]')
    st.markdown('#### :white[the ios is awesome]')
    st.markdown(
        '#### :white[chip and the performance of the processor is good]')

if selected == 'N-GRAMS':

    doc = df_2[(df_2['Ratings'] == 1) | (df_2['Ratings'] == 2)]

    col1, col2 = st.columns(2)
    with col1:
        n = st.selectbox(label="Select the word count",
                               options=(1, 2, 3, 4, 5), index=2)

    with col2:
        m = st.selectbox(label="Maximum list",
                         options=(10, 15, 20, 25, 30), index=2)

    df = dict_to_dataframe(doc['Processed_txt'], n, m)

    fig = px.bar(df, x="Words", y="Frequency")
    fig.update_layout(
        autosize=True,
        width=1200,
        height=400)
    st.markdown('#### :white[N-GRAMS FOR RATING 1 AND 2]')
    st.write(fig)

    doc = df_2[df_2['Ratings'] == 3]

    col1, col2 = st.columns(2)
    with col1:
        n = st.selectbox(label="Select the word counts",
                               options=(1, 2, 3, 4, 5), index=1)

    with col2:
        m = st.selectbox(label="Maximum lists",
                         options=(10, 15, 20, 25, 30), index=2)

    df = dict_to_dataframe(doc['Processed_txt'], n, m)

    fig = px.bar(df, x="Words", y="Frequency")
    fig.update_layout(
        autosize=True,
        width=1200,
        height=400)
    st.markdown('#### :white[N-GRAMS FOR RATING 3]')
    st.write(fig)

    doc = df_2[df_2['Ratings'] == 4]

    col1, col2 = st.columns(2)
    with col1:
        n = st.selectbox(label="Select word count",
                               options=(1, 2, 3, 4, 5), index=1)

    with col2:
        m = st.selectbox(label="Maximum order",
                         options=(10, 15, 20, 25, 30), index=2)

    df = dict_to_dataframe(doc['Processed_txt'], n, m)

    fig = px.bar(df, x="Words", y="Frequency")
    fig.update_layout(
        autosize=True,
        width=1200,
        height=400)
    st.markdown('#### :white[N-GRAMS FOR RATING 4]')
    st.write(fig)

    doc = df_2[df_2['Ratings'] == 5]

    col1, col2 = st.columns(2)
    with col1:
        n = st.selectbox(label="Select words count",
                               options=(1, 2, 3, 4, 5), index=1)

    with col2:
        m = st.selectbox(label="Maximum orders",
                         options=(10, 15, 20, 25, 30), index=2)

    df = dict_to_dataframe(doc['Processed_txt'], n, m)

    fig = px.bar(df, x="Words", y="Frequency")
    fig.update_layout(
        autosize=True,
        width=1200,
        height=400)
    st.markdown('#### :white[N-GRAMS FOR RATING 5]')
    st.write(fig)


if selected == 'PREDICTION':
    st.markdown("## :white[SAMPLE DATA]")
    st.table(df.head(5))

    pipe_1 = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                       ('clf', MultinomialNB(alpha=2.8954)),
                       ])
    pipe_1.fit(X_train, y_train)
    txt = st.text_input(label='Enter the Review')
    if st.button('PREDICT'):
        pred = pipe_1.predict([txt])
        st.markdown(f'#### :white[{(pred[0])}]')
