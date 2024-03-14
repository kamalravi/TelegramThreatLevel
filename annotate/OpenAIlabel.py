import openai
import pandas as pd

# Define your OpenAI API key
openai.api_key = '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'

Classes = f"Class 1- No Threat or Ambiguous Threat: Statements that do not contain any indication, suggestion, or desire for physical harm, imprisonment, or threat towards another person, group, or organization. This category represents comments that are not threatening in nature and do not pose any risk or harm to individuals or groups. Or Statements lacking clarity or specificity in expressing harm, leaving room for interpretation. These comments may contain language or context that could be interpreted as threatening such as defamation, name-calling, slandering but do not explicitly call for or suggest physical harm, imprisonment, or other forms of threat. \n\n Class 2- Judicial Threat: Statements explicitly calling for or threatening legal action, such as civil action, arrest, or criminal prosecution, within standard legal norms. While they may involve consequences within legal norms, they still signify a significant level of threat and potential legal ramifications. \n\n Class 3 - Non-judicial Threat: Statements explicitly advocating for non-legal actions or harm, such as physical violence or vigilante justice. It represents the most severe form of threat as it explicitly calls for harm or unlawful actions. If the statement also has a judicial threat, bel it non-judicial as it is extreme.'\n\n"

def classify_text(text):
    
    prompt = "Classify the following statement into one of the 3 classes:\n\n" + text + "\n\n{Classes}:"

    response = openai.Completion.create(
        engine="davinci-002",
        prompt=prompt,
        max_tokens=20,
        n=1,
        stop=None,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    
    # Extract the predicted class from the OpenAI API response
    predicted_class1 = response.choices[0].text.strip()
    predicted_class = response.choices[0].text.strip().split(") ")[0]

    print(predicted_class1)
    
    return predicted_class


# Example usage
inputText = "This criminal shoud hang" #"Wow! That must have beenna long distance call, from GITMO ??"
predicted_class = classify_text(inputText)
print("Predicted Class:", predicted_class)

# # Load the JSON file with texts
# df = pd.read_csv('/home/ravi/raviProject/DATA/Annotate/LabelV1/v1labeldf_V1.csv')

# # Apply the classify_text function to each text in the 'reply' column
# df['LabelOA'] = df['text'].apply(classify_text)

# # Save the results to a new file
# df.to_csv('/home/ravi/raviProject/DATA/Annotate/LabelV1/V1LabelBy3_V1.csv', index=False)
