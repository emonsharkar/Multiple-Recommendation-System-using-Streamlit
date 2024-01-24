import streamlit as st
import pandas as pd
import spacy

# Function to load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to recommend books or movies based on user input
def recommend(data, user_input, category, nlp, text_column):
    # Transform user input into a format suitable for recommendation
    user_document = f"{user_input['Present Mood']} {user_input['Past Mood']} {user_input['Genre']} {user_input['Language']}"
    user_tokens = [token.text.lower() for token in nlp(user_document)]

    # Find similar items using spaCy's word vectors
    similar_items = data.apply(lambda row: nlp(row[text_column]).similarity(nlp(" ".join(user_tokens))), axis=1)
    recommended_indices = similar_items.sort_values(ascending=False).head(5).index
    recommendations = data.loc[recommended_indices]

    return recommendations

# Streamlit app
st.title('Mood Boost')

# Page selection
page_selection = st.radio("Select an option:", ["Recommend Book", "Recommend Movie"])

# Load data
if page_selection == "Recommend Book":
    st.subheader('Book Recommendation')
    data = load_data('books.csv')
    text_column = 'Book'
elif page_selection == "Recommend Movie":
    st.subheader('Movie Recommendation')
    data = load_data('movies.csv')
    text_column = 'title'

# Load spaCy's English word vectors
nlp = spacy.load("en_core_web_sm")

# User input
mood_options = ["sad", "happy", "normal", "bored", "angry"]
genre_options = data['Genre'].unique().tolist() if page_selection == 'Recommend Book' else []
language_options = data['original_language'].unique().tolist() if page_selection == 'Recommend Movie' else []

user_input = {
    'Present Mood': st.selectbox('How do you feel right now?', mood_options),
    'Past Mood': st.selectbox('How was your mood 5 minutes ago?', mood_options),
    'Genre': st.selectbox('Select your preferred genre:', genre_options) if genre_options else None,
    'Language': st.multiselect('Select your preferred language(s):', language_options) if language_options else None
}

# Submit button
if st.button('Submit'):
    st.subheader('Recommendations:')
    recommendations = recommend(data, user_input, page_selection, nlp, text_column)
    for i, (_, item_data) in enumerate(recommendations.iterrows(), start=1):
        st.subheader(f'Recommended {page_selection} - {i}:')
        if page_selection == "Recommend Book":
            st.write(f"Name: {item_data['Book']}")
            st.write(f"Author: {item_data['Author(s)']}")
            st.write(f"Language: {item_data['Original language']}")
            st.write(f"First Publication: {item_data['First published']}")
            st.write(f"Genre: {item_data['Genre']}")
        elif page_selection == "Recommend Movie":
            st.write(f"Title: {item_data['title']}")
            st.write(f"Overview: {item_data['overview']}")
            st.write(f"Original Language: {item_data['original_language']}")
        st.write('')