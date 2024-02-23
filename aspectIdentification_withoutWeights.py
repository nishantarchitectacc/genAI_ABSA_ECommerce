import pandas as pd
import requests
import json
import re


raw_df = pd.read_csv("data/mergedData.csv")
feedbackAspects = []
for comment in raw_df['REVIEW_COMMENTS']:
    api_url = "http://127.0.0.1:5000/api/llama2"
    prompt = '''You are a feedback analyst who reviews customer feedback and responds with aspects.
    The response should follow below rules- 
    - If there is only 1 aspect identified, just respond with aspect. No explanation required, only aspect is required in response.
    - Aspect should be only be in 1 or 2 words not more than that.
    - If there are multiple aspects identified within single feedback then their Aspects should be comma "," seperated. 
    
    Above rules should be followed with no exception. 
    Below is the customer feedback- \n
    '''
    #usr_message = comment
    usr_message = comment
    body = {
            "user_prompt": prompt,
            "user_message": usr_message
            }
    response = requests.post(api_url, json=body)
    formatted_response = response.json()
    #sentiment = formatted_response['content'][0]['text'].strip(" ")
    sentiment = formatted_response['content'][0]['text']
    print(sentiment)
    print("-"*50)
    feedbackAspects.append(sentiment)

raw_df["FEEDBACK_ASPECTS"] = feedbackAspects
raw_df.to_csv("data/final_merged_data_withAspectswithoutWeights.csv")
