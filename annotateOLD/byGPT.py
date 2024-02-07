import requests

# Set up the API endpoint
url = 'https://api.openai.com/v1/engines/davinci-codex/completions'

# Set your OpenAI API key
api_key = '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'

# Set the input prompt for classification
input_prompt = "classify the input text into one of these seven classes: 1. Claims of illegal, criminal actions or behavior: Statements that allege or assert that a person, persons, or organization committed or is engaging in illegal or criminal activities without explicitly calling for arrest, prosecution, punishment, harm, or other action. 2. Claims of immoral or evil behaviors: Statements that allege, assert, or condemn a person, persons, or organization is engaged in immoral or evil behavior without explicitly calling for arrest, prosecution, punishment, harm, or other action. 3.Battle cries: Expressions of rallying call or motivating language, often associated with a cause or movement, but without explicit calls for arrest, prosecution, punishment, harm, or other action. 4. Implied support or ambiguous call or threats for action orharm: Statements that suggest or imply support for arrest, prosecution, punishment, or other harmful or threatening actions, but without explicit instructions or clear intent; statements may be in the form of rhetorical questions. 5.Calls or threats for judicial action or harm: Statements that explicitly call for or threaten legal action against a person, persons, or organization in the form of civil action, arrest, or criminal prosecution.  May include calls for punishment within standard legal norms, including prison, while excluding excessive, or cruel and unusual punishments. 6.Calls or threats for non-judicial or extra-judicial action or harm: Statements that explicitly call for or threaten non-judicial or extra-judicial action or harm against a person, persons, or organization, such as physical violence or vigilante justice. This includes actions or punishments outside  standard legal norms for being considered to be excessive, disproportionate, or cruel and unusual. 7.None of the above: No indication, suggestion, or desire of physical or emotional harms or threat towards another person, persons, or organization."

# Set the input text you want to classify
input_text = "We 1984 now"

# Set the model parameters
parameters = {
    'prompt': input_prompt + input_text,
    'max_tokens': 1,
    'top_p': 1.0,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0,
    'temperature': 0.0,
    'stop': '\n'
}

# Set the headers (include your API key)
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + api_key
}

# Make the API request
response = requests.post(url, json=parameters, headers=headers)

# Get the API response
data = response.json()

print(data)
# Get the predicted threat level index
predicted_index = data['choices'][0]['text']

# Print the predicted threat level label
print("Predicted Threat Level:", predicted_index)
