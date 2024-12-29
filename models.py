import openai
from dotenv import load_dotenv
import os

load_dotenv()   # Load environment variables from .env file
open_ai_key = os.getenv('OPENAI_API_KEY')
openai.api_key = open_ai_key

def get_recommendations(continent):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        engine="text-davinci-003",
        prompt=f"Recommend travel destinations in {continent}.",
        max_tokens=100
    )
    recommendations = response.choices[0].text.strip()
    return recommendations.split('\n')