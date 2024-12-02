import json
from groq import Groq

api_key = "<APIKEY>"
client = Groq(api_key=api_key)

# Function to process the scoreboard data
def process_scoreboard(scoreboard_data):
    prompt = f"""
    You will be given an improperly parsed cricket scoreboard. Your job is to extract the following information from it and return as JSON. 
    "{{ overs: , min_remaining: }}". Sometimes you will encounter some proxy for these things such as time remaining or overs remaining. Sometimes 
    you will not see overs explicitly labelled. You are to assume the most probable entry that corresponds to overs. Give only the json no extra verbosity.

    Here are some examples for you :

    SCB: "35/0> BURNS 20 %SIBLEY 15 3BETHELL 0-4-14SURREY9.4Run Rate 3.62Target 890000<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>"
    OUT: "{{overs: 9.4, min_remaining: None}}"

    SCB: "ECB36/1ROOT 15/9NORTHEAST77FISHER 0-15.5RAMORGAN14.5Run Rate 2.43Min Remaining Today 81.110030<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>"
    OUT: "{{overs: 14.5, min_remaining:81.11}}"

    SCB: "{scoreboard_data}"
    OUT:
    """
    
    print(prompt)

    # Prepare the data for the API call
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
        stream=False,
    )

    # Extract the result
    response = chat_completion.choices[0].message.content
    return response

# Load the JSON file
with open("/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/raw/scoreboards/rZKg8c8pW40_scb.json", 'r') as file:
    data = json.load(file)

# Check if the file has at least 40 entries
if len(data) >= 40:
    scoreboard_data = data[39]  # Extract the 40th entry (index 39)
    scb = scoreboard_data['scb']["<OCR>"]
    
    # Process the 40th entry
    result = process_scoreboard(scb)
    print(result)
else:
    print("The file does not contain 40 entries.")
