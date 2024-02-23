import pandas as pd
import requests
import json
import re


raw_df = pd.read_csv("data/mergedData.csv")
feedbackAspects = []
for comment in raw_df['REVIEW_COMMENTS']:
    api_url = "http://127.0.0.1:5000/api/llama2"
    prompt = '''You are a feedback analyst who reviews customer feedback and responds with aspects & it's weight. Please do not respond anything else other than Json.
    The response should be in json format following below rules- 
    - Json response should only have 2 keys: Aspect & Weight.
    - Aspect should be only be in 1 or 2 words not more than that.
    - Weight should be rated between 0 to 10, 0 being the lowest and 10 being the highest.
    - Keys should have identified Aspects and its Value which is weight
    - If there are multiple aspects identified within single feedback then their Aspects should be appended as "," seperated.
    - Respond with only Json output and no sentences prefixing Json. 

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
    feedbackAspects.append(sentiment)

raw_df["FEEDBACK_ASPECTS"] = feedbackAspects
raw_df.to_csv("data/final_merged_data_withAspectswithWeights.csv")

