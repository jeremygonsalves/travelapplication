import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from models import get_recommendations

# Title and info section
st.title('✈️ Travel Companion')
st.info('This is an app for custom implementation for travel recommendations')

# Collecting user preferences
st.header('Tell us about your travel preferences')


# Add 'Any' to the options list to make it a valid default option
continent = st.selectbox('Select Continent', ['Any', 'Europe', 'North America', 'Asia', 'Australia'])
continents = st.multiselect('Select Continents', ['Any', 'Europe', 'North America', 'Asia', 'Australia'], default=['Any'])

# Fetch destinations based on selected continents
def fetch_destinations(continents):
    if 'Any' in continents:
        return df['destination'].unique()
    else:
        return get_recommendations(continents)

destination_options = fetch_destinations(continents)
destination = st.selectbox('Preferred Destination', ['Any'] + list(destination_options))
travel_type = st.selectbox('What type of travel experience are you looking for?',
                           ['Adventure', 'Relaxation', 'Culture', 'Nature', 'Beach', 'City', 'Other'])

budget = st.slider('What is your travel budget?',
                   min_value=100,
                   max_value=5000,
                   value=1500,
                   step=100)

duration = st.slider('How many days are you planning to travel?',
                     min_value=1,
                     max_value=30,
                     value=7,
                     step=1)

# Dummy data for recommendations (this can later be replaced with real data and ML models)
travel_data = {
    'destination': ['Paris', 'New York', 'Tokyo', 'London', 'Barcelona', 'Sydney', 'Bali'],
    'type': ['Culture', 'City', 'Adventure', 'Culture', 'Beach', 'Adventure', 'Nature'],
    'budget': [2000, 3000, 1500, 2500, 1800, 2200, 1300],
    'duration': [5, 7, 10, 5, 7, 8, 6],
    'activities': ['Museums, Eiffel Tower', 'Broadway, Central Park', 'Hiking, Mount Fuji', 'Big Ben, Museums', 
                   'Beaches, Gaudi Architecture', 'Great Barrier Reef, Hiking', 'Beaches, Temples']
}

# Convert dummy data into a DataFrame
df = pd.DataFrame(travel_data)

# Function to recommend travel based on input data
def recommend_travel(user_data, df):
    # Filter destinations based on user input
    filtered_data = df[
        (df['type'] == user_data['travel_type']) &
        (df['budget'] <= user_data['budget']) &
        (df['duration'] >= user_data['duration'] - 2) &
        (df['duration'] <= user_data['duration'] + 2)
    ]
    
    # Use cosine similarity (optional)
    # Convert budget and duration to feature vectors and calculate similarity
    features = filtered_data[['budget', 'duration']].values
    user_features = np.array([user_data['budget'], user_data['duration']]).reshape(1, -1)
    scaler = StandardScaler().fit(features)
    features_scaled = scaler.transform(features)
    user_features_scaled = scaler.transform(user_features)
    
    similarity_scores = cosine_similarity(user_features_scaled, features_scaled)
    filtered_data['similarity'] = similarity_scores.flatten()
    
    # Sort by similarity and return top recommendations
    recommendations = filtered_data.sort_values(by='similarity', ascending=False).head(3)
    
    return recommendations

# Collect user input into a dictionary
user_data = {
    'destination': destination,
    'travel_type': travel_type,
    'budget': budget,
    'duration': duration
}

# Generate recommendations
if st.button('Get Travel Recommendations'):
    recommendations = recommend_travel(user_data, df)
    
    st.subheader('Recommended Travel Destinations:')
    
    for index, row in recommendations.iterrows():
        st.write(f"**Destination:** {row['destination']}")
        st.write(f"**Type:** {row['type']}")
        st.write(f"**Budget:** ${row['budget']}")
        st.write(f"**Duration:** {row['duration']} days")
        st.write(f"**Activities:** {row['activities']}")
        st.write('---')
