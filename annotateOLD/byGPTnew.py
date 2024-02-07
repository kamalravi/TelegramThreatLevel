import os
import openai

# Set your OpenAI API key
api_key = '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'

openai.api_key = api_key

# Set the input prompt for classification
classes = "1. Claims of illegal/criminal actions/behavior: Statements that allege or assert that a person, persons, group, or organization committed or is engaging in illegal or criminal activities without explicitly calling for arrest, prosecution, punishment, harm, or other action. This category excludes particularly heinous crimes like child abuse, molestation, sex- ual assault/rape, sex trafficking, torture, terrorism, murder, and mass murder/genocide. \
2. Claims of immoral/evil behaviors: Statements that assert that a person, persons, group, or organization is immoral or evil, and statements that allege or assert that a person, persons, group, or organization are committing acts or engaged in behavior that is considered particularly immoral or evil. Examples include grooming and pedophilia, in addition to particularly heinous crimes like child abuse, molestation, sexual assault/rape, sex trafficking, torture, terrorism, murder, and mass murder/genocide. \
3. Implied support/ambiguous call/threats for action/harm: Statements that suggest or imply support for arrest, prosecu- tion, punishment, or other harmful or threatening actions, but without explicit instructions or clear intent. Such implied or ambiguous statements are often in the form of rhetorical ques- tions or make reference to violence or harmful action through double meanings. This category can include protest/political chants and battle cries (e.g., 1776!, Live free or die!). \
4. Calls/threats for judicial action/harm: Statements that explicitly call for or threaten legal action against a person, persons, group, or organization in the form of civil action, arrest, or criminal prosecution. May include calls for punishment within standard legal norms, including prison, while excluding excessive, or cruel and unusual pun- ishments. This category includes statements in the form of protest/political chants (e.g. Lock her up). \
5. Calls/threats for non-judicial or extra-judicial action/harm: Statements that explicitly call for or threaten non-judicial or extra-judicial action or harm against a person, persons, groups, or organization, such as physical violence or vigilante justice. This includes actions or punishments considered outside of standard legal norms for being excessive, disproportionate, or cruel and unusual. This category also includes battle cries (e.g. Time to start a civil war!) and protest/political chants (e.g. Hang Mike Pence!) that refer to non-judicial or extra-judicial action. \
6. None of the above: No indication, suggestion, or desire of physical harm, im- prisonment, or threat towards another person, persons, group, or organization."

# Set the input text you want to classify
input_text = "We 1984 now"

response = openai.Completion.create(
  model= "gpt-3.5-turbo", #"text-davinci-003",
  prompt=f"Classify the threat level of the following statement:\n\n\"{input_text}\"\n\n{classes}",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

predicted_level = response.choices[0].text.strip()
print("Predicted Threat Level:", predicted_level)
