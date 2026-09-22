import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = pickle.load(open(r"C:\Users\Sai\A in Acodes\data science\projects\Customer Feedback Analysis\restaurent.pkl", 'rb'))
cv = pickle.load(open(r"C:\Users\Sai\A in Acodes\data science\projects\Customer Feedback Analysis\vectorizer.pkl", 'rb'))

# Page config
st.set_page_config(page_title="Restaurant Review Sentiment Analysis", page_icon="🍽️")

st.title("🍴 Restaurant Review Sentiment Analyzer")
st.write("Enter a customer review to see if it’s **Positive** or **Negative**.")

# Input box
user_review = st.text_area("Enter your review here:")

# Predict button
if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("⚠️ Please enter a review before predicting.")
    else:
        # Text cleaning (same as training)
        ps = PorterStemmer()
        review = re.sub('[^a-zA-Z]', ' ', user_review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)

        # Transform using TF-IDF
        review_tfidf = cv.transform([review]).toarray()

        # Predict
        prediction = model.predict(review_tfidf)[0]

        # Output
        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")