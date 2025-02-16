# Website Review Analysis Dashboard

## Overview

This dashboard is a powerful, end-to-end solution that evaluates website performance by analyzing user comments and reviews. Built using Streamlit, it leverages advanced natural language processing and machine learning techniques to extract sentiment, detect underlying emotions, filter spam, and visualize important trends through word clouds. The insights gained help drive customer-focused improvements, ensuring your website remains engaging and reliable.

## Key Benefits

*   Comprehensive Sentiment Analysis: User comments are accurately categorized based on their sentiment (positive, negative, or neutral) to provide a holistic understanding of customer satisfaction.
*   Nuanced Emotion Detection: Specific emotions expressed in user feedback are identified, going beyond sentiment to enable a deeper understanding of customer engagement and potential areas of emotional resonance.
*   Reliable Spam Prediction: Genuine customer feedback is automatically distinguished from spam, ensuring the accuracy and reliability of insights derived from user data.
*   Targeted Sentiment Filtering: Focused analysis and identification of specific strengths and weaknesses are facilitated by conveniently filtering user comments based on sentiment.
*   Holistic Website Rating: An aggregate rating for the website is computed based on user sentiments, providing a clear and easily trackable metric of overall performance.
*   Insightful Word Cloud Visualization: Dynamic word clouds are generated to visually highlight the most frequently used terms in user comments, revealing key topics and trends.

## Data Analytics Process

Here's a concise breakdown of the data analytics process for this project:

### Data Analytics Process (6 Key Steps):

1.  **Data Collection:**
    *   Web scraping (using `requests` and `beautifulsoup4`) to gather user reviews from platforms like Yelp.
    *   Aggregation of multiple datasets (e.g., TripAdvisor, YouTube comments) into a unified format.

2.  **Data Cleaning:**
    *   Preprocessing text by removing URLs, HTML tags, and special characters with `re` and `nltk`.
    *   Standardizing text (lowercasing, lemmatization) and filtering stopwords.

3.  **Data Analysis:**
    *   **Sentiment Analysis:** Fine-tuning a RoBERTa transformer model (`transformers` library) to classify comments as positive/neutral/negative (87% accuracy).
    *   **Emotion Detection:** Training a bidirectional LSTM model (`TensorFlow/Keras`) to identify emotions like joy, anger, or sadness (91% accuracy).
    *   **Spam Detection:** Implementing a Naive Bayes classifier (`scikit-learn`) with TF-IDF vectorization to filter spam (93% accuracy).

4.  **Data Visualization:**
    *   Generating interactive charts (`Plotly`) to display sentiment/emotion distributions.
    *   Creating word clouds (`wordcloud` library) to highlight frequent keywords.
    *   Calculating and visualizing an aggregate website rating.

5.  **Data Storytelling:**
    *   Integrating all components into a Streamlit dashboard for interactive exploration.
    *   Enabling sentiment filtering and real-time analysis of user feedback.

6.  **Deployment:**
    *   Packaging models (`Pickle`) and deploying the dashboard on Streamlit for user accessibility.

This pipeline transforms raw user comments into actionable insights, leveraging NLP and machine learning to evaluate website performance holistically.

## Core Features Details

This section provides detailed information on each of the core features of the dashboard:

### 1. Sentiment Analysis

*   **Area of Exploration:** The focus was on leveraging advanced natural language processing (NLP) techniques to analyze and classify the sentiment (positive, neutral, or negative) expressed in user comments. The challenge was to process unstructured textual data at scale while maintaining high accuracy.
*   **Method Implemented:** A fine-tuned RoBERTa model was used as the backbone for sentiment classification. The model was further enhanced with custom attention layers to better capture contextual relationships within the text. Transfer learning was applied by fine-tuning RoBERTa on a labeled dataset of user comments. Preprocessing steps included tokenization using Hugging Face's `transformers` library and padding/truncation to ensure uniform input length. The model output probabilities for each sentiment class, which were then converted into final predictions.
*   **Key Findings:** The fine-tuned RoBERTa model achieved an evaluation accuracy of 87% on the test dataset, demonstrating its ability to generalize well to unseen user comments. Attention layers improved the model's ability to focus on key parts of the text, such as adjectives and phrases that strongly indicate sentiment (e.g., "great service" or "terrible experience"). Preprocessing steps like lowercasing and removing special characters were critical in reducing noise and improving classification performance.

### 2. Emotion Detection

*   **Area of Exploration:** The goal was to detect and classify specific emotions (e.g., joy, sadness, anger, fear) in user feedback to gain deeper insights into customer engagement and satisfaction levels. This required handling complex emotional nuances in textual data.
*   **Method Implemented:** A deep learning model was built using LSTM layers with bidirectional architecture to capture both forward and backward dependencies in text. The model was trained on a labeled dataset of emotions using Keras with TensorFlow as the backend. Preprocessing steps included tokenization, padding, and truncation to ensure uniform input length. The output layer used a softmax activation function to predict probabilities for each emotion class.
*   **Key Findings:** The model achieved an evaluation accuracy of 91%, effectively capturing emotional undertones in user comments. Bidirectional LSTMs improved the model's ability to understand contextual relationships in sentences, especially for complex emotions like "surprise" or "fear." Preprocessing steps like stopword removal and lemmatization were critical for improving model performance by reducing noise in the data.

### 3. Spam Prediction

*   **Area of Exploration:** The focus was on identifying spam comments within user feedback to ensure that only authentic data contributes to insights. This required distinguishing between genuine reviews and spam efficiently.
*   **Method Implemented:** A Naive Bayes classifier was trained on TF-IDF vectorized text data. The TF-IDF vectorizer converted raw text into numerical features by calculating term frequency-inverse document frequency scores for each word. The model was trained on a labeled dataset of YouTube comments containing both spam and non-spam entries.
*   **Key Findings:** The Naive Bayes classifier achieved an evaluation accuracy of 93%, demonstrating its effectiveness for binary classification tasks like spam detection. TF-IDF vectorization successfully captured the importance of words like "free," "click," or "subscribe," which are strong indicators of spam. Preprocessing plays a crucial role in spam detection; removing irrelevant noise (e.g., HTML tags) enhances model accuracy.

### 5. Website Rating

*   **Area of Exploration:** Developing an aggregate rating system based on sentiment analysis results was explored to provide a high-level metric summarizing website performance.
*   **Method Implemented:** An aggregate rating was computed by assigning numerical scores to each sentiment category (e.g., +1 for positive, 0 for neutral, -1 for negative). The average score across all user comments was calculated and scaled to a 5-point rating system. This rating was displayed prominently on the dashboard as a key performance indicator (KPI).
*   **Key Findings:** The aggregate rating provided an intuitive summary of overall customer sentiment towards the website. Positive ratings correlated strongly with high proportions of positive comments, validating the effectiveness of this metric as a KPI. Scaling the rating system ensured compatibility with common review platforms like Yelp or Google Reviews.

### 6. WordCloud Visualization

*   **Area of Exploration:** Visualizing frequently mentioned terms in user comments was explored as a way to identify key topics and trends quickly.
*   **Method Implemented:** Word clouds were generated using the `wordcloud` library after preprocessing text data. Stopwords were removed, and words were weighted based on their frequency in user comments. RAKE (Rapid Automatic Keyword Extraction) was also used to identify multi-word phrases that appeared frequently.
*   **Key Findings:** Word clouds effectively highlighted recurring themes such as "service," "quality," or "delivery," providing quick insights into what customers care about most. Keywords extracted using RAKE added depth by identifying phrases rather than single words (e.g., "customer support" instead of just "support").

## Core Libraries Used in the Project

*   Streamlit: Builds the interactive dashboard.
*   Pandas & NumPy: Data manipulation and numerical analysis.
*   Scikit-learn: Model building (Naive Bayes), feature extraction (TF-IDF), and evaluation.
*   nltk & re: Text preprocessing and cleaning.
*   TensorFlow & Keras: Creating and training the emotion detection model.
*   Transformers & Torch: Leveraging pre-trained RoBERTa for sentiment analysis.
*   BeautifulSoup4 & Requests: Web scraping for data collection.
*   Plotly & Matplotlib: Data visualization and interactive charts.
*   WordCloud & rake_nltk: Generating word clouds and extracting keywords.
*   Pickle: Model serialization and persistence.

