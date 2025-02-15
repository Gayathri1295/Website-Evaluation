import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from rake_nltk import Rake
import plotly.express as px
import plotly.graph_objects as go
import re
import nltk
import tensorflow as tf
import numpy as np
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
import json
import torch
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

from bs4 import BeautifulSoup
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Set page config for wide layout
st.set_page_config(page_title="Emotion Detection and Sentiment Analysis", page_icon=":bar_chart:", layout="wide")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')  # For POS tagging

# Initialize NLTK tools
stop_words = set(stopwords.words('english')).union({
    'would', 'could', 'should', 'get', 'us', 'go', 'even', 'really', 'make', 'made', 'many'
})


# Load the trained model and TF-IDF vectorizer
with open("model.pkl", "rb") as model_file:
    model_spam = pickle.load(model_file)

with open("tfidf-vect.pkl", "rb") as tfidf_vect_file:
    tfidf_vect = pickle.load(tfidf_vect_file)

# Initialize session state
if 'emotions_count' not in st.session_state:
    st.session_state['emotions_count'] = None

if 'sentiments_count' not in st.session_state:
    st.session_state['sentiments_count'] = None

# Load the emotion detection model and tokenizer
emotion_model = load_model('Emotion_Detection.h5')
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    data = f.read()
    emotion_tokenizer = tokenizer_from_json(data)

# Load the sentiment analysis model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model.eval()

# Define the scrape_comments function
def scrape_multiple_pages(url):
    comments_data = []

    # Extract base URL and pagination parameters
    base_url, params = url.split("?")[0], url.split("?")[1]

    # Extract the current start value
    start_param = re.search(r'start=(\d+)', params)
    if start_param:
        start_value = int(start_param.group(1))
    else:
        print("Start parameter not found in URL.")
        return None

    # Generate URLs for the next 5 pages
    for i in range(1, 6):  # Starts from 1 to scrape 5 more pages
        new_start = start_value + i * 10
        new_url = f"{base_url}?{params.replace(start_param.group(0), 'start=' + str(new_start))}#reviews"
        response = requests.get(new_url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            comment_elements = soup.find_all(class_=re.compile(r'\bcomment__.*'))

            if comment_elements:
                for element in comment_elements:
                    comment_text = element.find('span', class_='raw__09f24__T4Ezm').text.strip()
                    comments_data.append({"Comment": comment_text})
            else:
                print(f"No comments found on page {i}.")

    comments_df = pd.DataFrame(comments_data)
    return comments_df


def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = re.sub(r'\S*@\S*\s?|<.*?>|[^a-zA-Z\s]|[\xc3\xa1\xc3\xa9\xc3\xad\xc3\xb3\xc3\xba\xc3\xb1\xc3\x81\xc3\x89\xc3\x8d\xc3\x93\xc3\x9a\xc3\x91]', '', text)

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return ' '.join(tokens)

def predict_emotion(text):
    emotions_count = {'anger': 0, 'fear': 0, 'joy': 0, 'love': 0, 'sadness': 0, 'surprise': 0}
    sentences = text.split('\n')
    
    for sentence in sentences:
        if sentence.strip() == "":
            continue
        preprocessed_text = preprocess_text(sentence)
        sequence = emotion_tokenizer.texts_to_sequences([preprocessed_text])
        padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')
        
        prediction = emotion_model.predict(padded_sequence)
        predicted_index = np.argmax(prediction)
        emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
        
        predicted_emotion = emotions[predicted_index]
        emotions_count[predicted_emotion] += 1

    return emotions_count

def preprocess_text_before_tokenization(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    return text

def classify_spam(comment):
    # Preprocess the comment
    preprocessed_comment = preprocess_text_before_tokenization(comment)
    # Vectorize the preprocessed comment using the TF-IDF vectorizer
    comment_tfidf = tfidf_vect.transform([preprocessed_comment])
    # Predict spam classification using the trained model
    prediction = model_spam.predict(comment_tfidf)
    
    if prediction[0] == 1:
        spam_count = 1
        not_spam_count = 0
    else:
        spam_count = 0
        not_spam_count = 1

  # Count of comments predicted as not spam
    return spam_count, not_spam_count

def predict_sentiment(text):
    preprocessed_text = preprocess_text_before_tokenization(text)
    inputs = tokenizer.encode_plus(
        preprocessed_text, return_tensors='pt', max_length=512, truncation=True, padding='max_length', add_special_tokens=True
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    sentiment_scores = torch.softmax(logits, dim=1).detach().cpu().numpy()
    predicted_class_id = np.argmax(sentiment_scores, axis=1)[0]

    sentiments = ['Negative', 'Neutral', 'Positive']
    sentiment = sentiments[predicted_class_id]
    return sentiment

def clean_text(text):
    tokens = word_tokenize(text)
    # Filter non-alphabetic words and stopwords, and focus on adjectives and nouns
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    tagged_tokens = pos_tag(filtered_tokens)
    # Keep only nouns and adjectives
    descriptive_words = [word for word, tag in tagged_tokens if tag.startswith('NN') or tag.startswith('JJ')]
    return " ".join(descriptive_words)

def generate_word_cloud(text):
    # Preprocess text
    preprocessed_text = clean_text(text)
    # Initialize RAKE using preprocessed text
    rake_nltk_var = Rake()
    rake_nltk_var.extract_keywords_from_text(preprocessed_text)
    keyword_extracted = rake_nltk_var.get_ranked_phrases()
    # Join keywords for the word cloud
    text_for_wordcloud = " ".join(keyword_extracted)
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, height=700, background_color='white', margin=5
    ).generate(text_for_wordcloud)
    # Display using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(top=10, bottom=0.1, left=0.1, right=0.9)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def main():
    st.title("Website Evaluation - Sentiment Analysis")
    url_input = st.text_input("Enter Yelp website URL:")

    if st.button("Analyze", key="analyze_button"):
        if not url_input:
            st.warning("Please enter a valid URL.")
        else:
            comments_df = scrape_multiple_pages(url_input)
            if comments_df is not None:
                st.session_state['texts'] = comments_df['Comment'].tolist()
                st.session_state['emotions_count'] = predict_emotion('\n'.join(st.session_state['texts']))
                st.session_state['sentiments_count'] = {'Negative': 0, 'Neutral': 0, 'Positive': 0}
                st.session_state['spam_count'] = 0
                st.session_state['not_spam_count'] = 0

                for text in st.session_state['texts']:
                    if text.strip():
                        sentiment = predict_sentiment(text)
                        st.session_state['sentiments_count'][sentiment] += 1

                        # Classify comment as spam or not spam
                        spam_count, not_spam_count = classify_spam(text)
                        st.session_state['spam_count'] += spam_count
                        st.session_state['not_spam_count'] += not_spam_count


                st.session_state['analysis_done'] = True
            else:
                st.error("Failed to scrape comments. Check the URL and try again.")

    if st.session_state.get('analysis_done', False):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("**Emotion Analysis**")
            df_emotions = pd.DataFrame(list(st.session_state['emotions_count'].items()), columns=['Emotion', 'Count'])
            fig_emotions = px.pie(df_emotions, names='Emotion', values='Count',
                                  color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_emotions, use_container_width=True)
           
            # Display average rating inside a bordered container
            with st.container():
                column1, column2 = st.columns(2)
                
                with column1:
                    st.markdown(
                        f"""
                        <div style="border-width: 2px; border-style: solid; border-color: #ccc; padding: 10px; border-radius: 10px;">
                            <div class="column1">
                                <h3 style='margin-bottom: 0px; text-align: center;'>Average Rating</h3>
                                <p style='font-size: 24px; margin-bottom: 0px; text-align: center;'>{calculate_average_rating(st.session_state['sentiments_count']):.2f}/5</p>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with column2:
                    st.markdown(
                        f"""
                        <div style="border-width: 2px; border-style: solid; border-color: #ccc; padding: 10px; border-radius: 10px;">
                            <div class="column2">
                                <h3 style='margin-bottom: 0px; text-align: center;'>Total Comments</h3>
                                <p style='font-size: 24px; margin-bottom: 0px; text-align: center;'>{calculate_total_comments(st.session_state['sentiments_count'])}</p>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown('<style>.container { display: flex; justify-content: space-between; margin-bottom: 20px; }</style>', unsafe_allow_html=True)


            with st.container():
                st.markdown('<style>.container { display: flex; justify-content: space-between; margin-top: 50px; }</style>', unsafe_allow_html=True)
                st.subheader("Filter by sentiment")
                selected_sentiment = st.selectbox("Choose sentiment to display:", ["Positive", "Neutral", "Negative"])

                if selected_sentiment:
                    st.session_state['filtered_texts'] = filter_texts(selected_sentiment, st.session_state['texts'])

                    if selected_sentiment == "Positive":
                        shortened_texts = [text[:100] for text in st.session_state['filtered_texts'] if text.strip()]
                    elif selected_sentiment == "Negative":
                        shortened_texts = [text[:100] for text in st.session_state['filtered_texts'] if text.strip()]
                    else:
                        shortened_texts = [text[:100] for text in st.session_state['filtered_texts'] if text.strip()]


                    if st.session_state['filtered_texts']:
                        num_rows = len(st.session_state['filtered_texts'])
                        color_even = 'rgb(8, 81, 156)'
                        color_odd = 'rgb(49, 130, 189)'
                        colors = [color_even if i % 2 == 0 else color_odd for i in range(num_rows)]

                        fig = go.Figure(data=[go.Table(
                            header=dict(values=['<b>Texts Classified as ' + selected_sentiment + '</b>'],
                                        fill_color='grey',
                                        align='center',
                                        font=dict(color='white', size=16)),
                            cells=dict(values=[shortened_texts],
                                       fill_color=[[color_even, color_odd] * (num_rows // 2 + num_rows % 2)],
                                       align='left',
                                       font=dict(color='white', size=14),
                                       height=30)
                        )])
                        fig.update_layout(margin=dict(l=20, r=20, t=10, b=20), height = 500)
                        st.plotly_chart(fig, use_container_width=True, height = 500)
                    else:
                        st.write(f"No texts found for {selected_sentiment} sentiment.")

                st.markdown('<style>.container { display: flex; justify-content: space-between; }</style>', unsafe_allow_html=True)
                st.markdown('<style>.container + .container { margin-top: 20px; }</style>', unsafe_allow_html=True)


        with col2:
            st.subheader("**Sentiment Analysis**")
            df_sentiments = pd.DataFrame(list(st.session_state['sentiments_count'].items()), columns=['Sentiment', 'Count'])
            fig_sentiments = px.bar(df_sentiments, x='Sentiment', y='Count', color='Sentiment')
            st.plotly_chart(fig_sentiments, use_container_width=True)

            st.subheader("Spam Classification")
            df_spam = pd.DataFrame({'Category': ['Spam', 'Not Spam'], 'Count': [st.session_state['spam_count'], st.session_state['not_spam_count']]})
            fig_spam = px.bar(df_spam, x='Count', y='Category', orientation='h', 
                       color='Category', text='Count', 
                       labels={'Count': 'Count', 'Category': 'Category'}, 
                       color_discrete_sequence=['#636EFA', '#EF553B'])
            st.plotly_chart(fig_spam, use_container_width=True)

            st.subheader('Word Cloud')
            with st.container():
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.session_state['texts']:
                    fig = generate_word_cloud('\n'.join(st.session_state['texts']))
                    st.pyplot(fig)
                else:
                    st.warning("No comments available to generate word cloud.")


def calculate_average_rating(sentiments_count):
    total_count = sum(sentiments_count.values())
    total_score = sentiments_count['Negative'] * 1 + sentiments_count['Neutral'] * 3 + sentiments_count['Positive'] * 5
    if total_count == 0:
        return 0
    return total_score / total_count

def calculate_total_comments(sentiments_count):
    return sum(sentiments_count.values())

# Define a function to filter texts based on sentiment
def filter_texts(sentiment, texts):
    return [text for text in texts if predict_sentiment(text) == sentiment]

# Initialize session state variables
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False
if 'emotions_count' not in st.session_state:
    st.session_state['emotions_count'] = None
if 'sentiments_count' not in st.session_state:
    st.session_state['sentiments_count'] = None
if 'texts' not in st.session_state:
    st.session_state['texts'] = []
if 'selected_sentiment' not in st.session_state:
    st.session_state['selected_sentiment'] = None
if 'filtered_texts' not in st.session_state:
    st.session_state['filtered_texts'] = []

# Run the main function
if __name__ == "__main__":
    main()

