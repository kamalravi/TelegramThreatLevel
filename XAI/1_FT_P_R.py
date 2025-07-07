# %%
import numpy as np
import pandas as pd
import glob
import json
import math
from natsort import natsorted
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# # model predictions

# %%
df60k = pd.read_json('/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier/Labeled_2261_test_yPred_preTrainFT_RoBERTa_60kSteps.json', orient='records')

# %%
df60k.head(1)

# %%
df60k['y_pred'].value_counts()

# %%
# Print the first few rows to verify
df60k.head()

# %%
precision_score(df60k['Label'], df60k['y_pred'], average='weighted')

# %%
recall_score(df60k['Label'], df60k['y_pred'], average='weighted')

# %%
f1_score(df60k['Label'], df60k['y_pred'], average='weighted')

# %%
accuracy_score(df60k['Label'], df60k['y_pred'])

# %%
# Generate classification report
class_report = classification_report(df60k['Label'], df60k['y_pred'])

# %%
print("Classification Report:\n", class_report)

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

df = df60k.copy()

# Calculate the confusion matrix
conf_matrix = confusion_matrix(df['Label'], df['y_pred'])
# Calculate sums for each column and row
row_sums = conf_matrix.sum(axis=1)
col_sums = conf_matrix.sum(axis=0)
# Calculate required percentages for columns
col_percent = [
    [
        conf_matrix[i, i] / row_sums[i] * 100 if row_sums[i] > 0 else 0,  # TP percentage
        (row_sums[i] - conf_matrix[i, i]) / row_sums[i] * 100 if row_sums[i] > 0 else 0  # FP percentage
    ] for i in range(conf_matrix.shape[0])
]
# Calculate required percentages for rows
row_percent = [
    [
        conf_matrix[i, i] / col_sums[i] * 100 if col_sums[i] > 0 else 0,  # TP percentage
        (col_sums[i] - conf_matrix[i, i]) / col_sums[i] * 100 if col_sums[i] > 0 else 0  # FP percentage
    ] for i in range(conf_matrix.shape[0])
]
# Create a 4x4 matrix for the confusion matrix and the calculated percentages
sum_matrix = np.zeros((4, 4))  # Create a 4x4 matrix
# print(conf_matrix)
sum_matrix[:-1, :-1] = conf_matrix  # Top-left 3x3 for the confusion matrix
# print(sum_matrix)
# Fill last row and column with calculated percentages
for i in range(3):
    sum_matrix[i, -1] = col_percent[i][0]  # TP % in last column
    sum_matrix[i, -2] = col_percent[i][1]  # FP % in last column
    sum_matrix[-1, i] = row_percent[i][0]  # TP % in last row
    sum_matrix[-2, i] = row_percent[i][1]  # FP % in last row
# Calculate the sum for the entire confusion matrix
total_sum = conf_matrix.sum()
# Calculate the sum of the diagonal values (0,0), (1,1), (2,2)
diagonal_sum = conf_matrix[0, 0] + conf_matrix[1, 1] + conf_matrix[2, 2]
# Calculate the sum of all values except for the diagonal ones
off_diagonal_sum = total_sum - diagonal_sum
# Calculate the desired values for the bottom right cell
bottom_right_value_1 = diagonal_sum / total_sum
bottom_right_value_2 = off_diagonal_sum / total_sum
# print(sum_matrix)
# Create a figure with a specified width and height
# plt.figure(figsize=(3.5, 3.5))
plt.figure(figsize=(2.5, 2.5))
# Create a color array for the confusion matrix (4x4, each cell has 3 values for RGB)
colors = np.ones((4, 4, 3))  # Initialize all cells to white [1, 1, 1]
# Create the colormap from the colors array
cmap = plt.cm.colors.ListedColormap(colors.reshape(-1, 3))
# print(sum_matrix)
sum_matrix[:-1, :-1] = conf_matrix  # Top-left 3x3 for the confusion matrix
# print("afeter change")
# print(sum_matrix)
# Display the confusion matrix with percentages
plt.imshow(sum_matrix, cmap=cmap, aspect='auto')

# Add text for the confusion matrix and metrics
for i in range(4):
    for j in range(4):
        # Only display numeric values for the confusion matrix cells
        if i < 3 and j < 3:  # Confusion matrix cells
            color = 'green' if i == j else 'red'  # Set diagonal color to green
            plt.text(j, i, f"{sum_matrix[i, j]:.0f}", ha='center', va='center', color=color, fontsize=8)
        
        # True Positives (green) and False Positives (red) in last row (TP/FP for columns)
        elif i == 3 and j < 3:  # TP percentages in the last row
            plt.text(j, i-0.1, f"{row_percent[j][0]:.1f}%", ha='center', va='center', color='green', fontsize=8)
            plt.text(j, i + 0.15, f"{row_percent[j][1]:.1f}%", ha='center', va='center', color='red', fontsize=8)
        
        # True Positives (green) and False Positives (red) in the last column (TP/FP for rows)
        elif j == 3 and i < 3:  # Percentages in the last column (for rows)
            plt.text(j, i, f"{col_percent[i][0]:.1f}%", ha='center', va='bottom', color='green', fontsize=8)  # TP (upper line)
            plt.text(j, i+ 0.06, f"{col_percent[i][1]:.1f}%", ha='center', va='top', color='red', fontsize=8)  # FP (lower line)

        # True Positives (green) and False Positives (red) in the last cell (TP/FP for rows)
        elif j == 3 and i == 3:  # Percentages in the last column (for rows)
            plt.text(j, i, f"{bottom_right_value_1:.1f}%", ha='center', va='bottom', color='green', fontsize=8)  # TP (upper line)
            plt.text(j, i+ 0.06, f"{bottom_right_value_2:.1f}%", ha='center', va='top', color='red', fontsize=8)  # FP (lower line)

# Set ticks and labels
plt.xticks(ticks=np.arange(4), labels=['NT', 'JT', 'NJT', 'TP/FP'], fontsize=8)
plt.yticks(ticks=np.arange(4), labels=['NT', 'JT', 'NJT', 'TP/FP'], fontsize=8)
# Add borders
for i in range(4):
    for j in range(4):
        plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='black', facecolor='none', lw=0.5))
# Aesthetic customizations
# plt.gca().set_facecolor('#ffffff')  # White background for better contrast
plt.grid(False)  # Disable grid for a clean look
# Add labels
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# Show the plot
plt.tight_layout()

plt.savefig('/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier/ConfusionMatrixDisplay.png', format='png', dpi=1200)


plt.show()

print(sum_matrix)

# %% [markdown]
# # Misclassifications

# %%
# Filter misclassifications (where Label != y_pred)
df = df60k.copy()
misclassified_df = df[df['Label'] != df['y_pred']]
classified_df = df[df['Label'] == df['y_pred']]

# Display the misclassified DataFrame
print(misclassified_df.shape)

# %%
misclassified_df.to_csv('/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier/Labeled_2261_test_yPred_preTrainFT_RoBERTa_60kSteps_MisClas.csv')

# %%
classified_df.to_csv('/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier/Labeled_2261_test_yPred_preTrainFT_RoBERTa_60kSteps_Clas.csv')

# %% [markdown]
# # Prediction and Label bias comparison

# %%
# Grouping by 'telegramChannel'
df = df60k.copy()
result_df = df.groupby('telegramChannel').agg(
    no_of_unique_telegramChannel=('telegramChannel', 'size'),
    
    # Counts for y_pred (predictions)
    label_0_pred_count=('y_pred', lambda x: (x == 0).sum()),
    label_1_pred_count=('y_pred', lambda x: (x == 1).sum()),
    label_2_pred_count=('y_pred', lambda x: (x == 2).sum()),
    
    # Counts for Label (actual)
    label_0_actual_count=('Label', lambda x: (x == 0).sum()),
    label_1_actual_count=('Label', lambda x: (x == 1).sum()),
    label_2_actual_count=('Label', lambda x: (x == 2).sum())
).reset_index()

# Renaming columns to make it more readable
result_df.columns = ['telegramChannel', 'Sample count', 'No Threat (Pred)', 'Judicial Threat (Pred)', 'Non-Judicial Threat (Pred)',
                    'No Threat (Actual)', 'Judicial Threat (Actual)', 'Non-Judicial Threat (Actual)']

# Step 1: Remove '.json' from 'telegramChannel'
result_df['telegramChannel'] = result_df['telegramChannel'].str.replace('.json', '', regex=False)

# Step 2: Calculate percentages for 'No Threat', 'Judicial Threat', and 'Non-Judicial Threat' (Predictions)
result_df['No Threat (Pred %)'] = (result_df['No Threat (Pred)'] / result_df['Sample count'] * 100).round(2)
result_df['Judicial Threat (Pred %)'] = (result_df['Judicial Threat (Pred)'] / result_df['Sample count'] * 100).round(2)
result_df['Non-Judicial Threat (Pred %)'] = (result_df['Non-Judicial Threat (Pred)'] / result_df['Sample count'] * 100).round(2)

# Step 3: Calculate percentages for 'No Threat', 'Judicial Threat', and 'Non-Judicial Threat' (Actual labels)
result_df['No Threat (Actual %)'] = (result_df['No Threat (Actual)'] / result_df['Sample count'] * 100).round(2)
result_df['Judicial Threat (Actual %)'] = (result_df['Judicial Threat (Actual)'] / result_df['Sample count'] * 100).round(2)
result_df['Non-Judicial Threat (Actual %)'] = (result_df['Non-Judicial Threat (Actual)'] / result_df['Sample count'] * 100).round(2)

# Step 4: Create new columns for combined threat (Judicial + Non-Judicial) for both Pred and Actual
result_df['Combined Threat (Pred %)'] = result_df['Judicial Threat (Pred %)'] + result_df['Non-Judicial Threat (Pred %)']
result_df['Combined Threat (Actual %)'] = result_df['Judicial Threat (Actual %)'] + result_df['Non-Judicial Threat (Actual %)']

# Step 5: Sort by 'Combined Threat (Pred %)' in ascending order
result_df = result_df.sort_values(by='Sample count', ascending=True)

# Step 6: Keep only the relevant columns
result_df = result_df[['telegramChannel', 'Sample count', 
                       'No Threat (Pred %)', 'Judicial Threat (Pred %)', 'Non-Judicial Threat (Pred %)', 
                       'No Threat (Actual %)', 'Judicial Threat (Actual %)', 'Non-Judicial Threat (Actual %)']]

# Display the result
print(result_df)


# %% [markdown]
# # Precision-Recall Curve

# %%
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assuming df 
df = pd.read_json('/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier/Labeled_2261_test_yPred_preTrainFT_RoBERTa_withProbScores.json', orient='records')
y_true = df['Label']
y_probs = np.array(df['prob_scores'].tolist())  # Probabilities for each class (3 columns for 3 classes)

# Binarize the labels for multi-class Precision-Recall (One-vs-Rest strategy)
n_classes = 3
y_bin = label_binarize(y_true, classes=[0, 1, 2])

# Class names corresponding to the labels
class_names = {0: 'No Threat', 1: 'Judicial Threat', 2: 'Non-Judicial Threat'}

# Compute Precision-Recall and Average Precision for each class
precision = dict()
recall = dict()
average_precision = dict()

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
    average_precision[i] = average_precision_score(y_bin[:, i], y_probs[:, i])

# Set global font size to 8
plt.rcParams.update({'font.size': 8})

# Define line styles
line_styles = ['-', '--', ':']  # Solid, dashed, and dotted lines

# Plotting the Precision-Recall curves
plt.figure(figsize=(3.5, 2.5))
colors = cycle(['green', 'darkorange', 'red'])

for i, (color, line_style) in zip(range(n_classes), zip(colors, line_styles)):
    plt.plot(recall[i], precision[i], color=color, lw=2, linestyle=line_style,
             label='{} (AP = {:.2f})'.format(class_names[i], average_precision[i]))


print(average_precision)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.xlim([0.0, 1.03])
plt.ylim([0.28, 1.03])
plt.tight_layout()

plt.savefig('/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier/PrecisionRecallCurve.png', format='png', dpi=1200)

plt.show()

# %% [markdown]
# # Temporal robustness

# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from scipy.signal import savgol_filter

# Sample DataFrame (replace with your actual data)
df = df60k.copy()

# 1. Convert 'replyDate' from Unix timestamp to datetime
df['replyDate'] = pd.to_datetime(df['replyDate'], unit='s')

# 2. Extract month and year from 'replyDate'
df['month'] = df['replyDate'].dt.to_period('M')

# 3. Calculate F1 score (weighted) and count texts per month
monthly_f1_scores = []
monthly_counts = []
sorted_months = []

for month, group in df.groupby('month'):
    f1 = f1_score(group['Label'], group['y_pred'], average='weighted')
    monthly_f1_scores.append(f1)
    monthly_counts.append(len(group))
    sorted_months.append(month)

# 4. Sort the data by month in ascending order
sorted_indices = sorted(range(len(sorted_months)), key=lambda i: sorted_months[i])
sorted_months = [sorted_months[i] for i in sorted_indices]
monthly_f1_scores = [monthly_f1_scores[i] for i in sorted_indices]
monthly_counts = [monthly_counts[i] for i in sorted_indices]

# Convert sorted months to datetime for plotting
months = pd.to_datetime([str(m) for m in sorted_months])

# 5. Adjust the window size for Savitzky-Golay filter
# Apply Savitzky-Golay filter to smooth the F1 score
smoothed_f1_scores = savgol_filter(monthly_f1_scores, 5, 1)  # Window size, polynomial order
# smoothed_f1_scores = monthly_f1_scores

# 6. Plotting
fig, ax1 = plt.subplots(figsize=(5, 3))

# Plot the number of texts (left y-axis)
ax1.plot(months, monthly_counts, linestyle='--', color='#DC7633', label='# of Replies')
ax1.set_xlabel('Timeline')
ax1.set_ylabel('# of Replies', color='#DC7633')
ax1.tick_params(axis='y', labelcolor='#DC7633')

# 7. Create the second y-axis for F1 score
ax2 = ax1.twinx()
ax2.plot(months, smoothed_f1_scores, color='#558B2F', label='F1 Score')
ax2.set_ylabel('F1 Score', color='#558B2F')
ax2.tick_params(axis='y', colors='#558B2F')

# Ensure F1 score axis starts at 0 and does not show negative values
ax2.set_ylim(bottom=0)

# Add legends for both plots
ax1.legend(loc='lower right', fontsize=8)
ax2.legend(loc='upper right', fontsize=8)

# Add grid lines
ax1.grid(True, which='both', linestyle=':', linewidth=0.5)
ax2.grid(True, which='both', linestyle=':', linewidth=0.5)

# 8. Final formatting
# fig.suptitle('Monthly Number of Texts and Weighted F1 Scores (Smoothed and Original)')
fig.tight_layout()

plt.savefig('/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier/Temporal_F1_preTrainFT_RoBERTa_60kSteps.png', format='png', dpi=1200)

plt.show()

# %% [markdown]
# # External case study (OOD)

# %%
dfEX = pd.read_json('/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier/remaining_V7_1M_NotUsed_yPred_preTrainFT_RoBERTa.json', orient='records')

# %%
dfEX.shape

# %%
dfEX.head()

# %%
# Grouping by 'telegramChannel'
df = dfEX.copy()
result_df = df.groupby('telegramChannel').agg(
    no_of_unique_telegramChannel=('telegramChannel', 'size'),
    label_0_count=('y_pred', lambda x: (x == 0).sum()),
    label_1_count=('y_pred', lambda x: (x == 1).sum()),
    label_2_count=('y_pred', lambda x: (x == 2).sum())
).reset_index()

# Renaming columns to make it more readable
result_df.columns = ['telegramChannel', 'Sample count', 'No Threat', 'Judicial Threat', 'Non-Judicial Threat']

# Step 1: Remove '.json' from 'telegramChannel'
result_df['telegramChannel'] = result_df['telegramChannel'].str.replace('.json', '', regex=False)

# Step 2: Calculate percentages for 'No Threat', 'Judicial Threat', and 'Non-Judicial Threat'
result_df['No Threat (%)'] = (result_df['No Threat'] / result_df['Sample count'] * 100).round(2)
result_df['Judicial Threat (%)'] = (result_df['Judicial Threat'] / result_df['Sample count'] * 100).round(2)
result_df['Non-Judicial Threat (%)'] = (result_df['Non-Judicial Threat'] / result_df['Sample count'] * 100).round(2)

# Step 3: Create a new column for combined threat (Judicial + Non-Judicial)
result_df['Combined Threat (%)'] = result_df['Judicial Threat (%)'] + result_df['Non-Judicial Threat (%)']

# Step 4: Sort by 'Non-Judicial Threat (%)' in ascending order
result_df = result_df.sort_values(by='Combined Threat (%)', ascending=True)

# Step 5: Keep only the relevant columns
result_df = result_df[['telegramChannel', 'Sample count', 'No Threat (%)', 'Judicial Threat (%)', 'Non-Judicial Threat (%)', 'Combined Threat (%)']]

# Display the result
print(result_df)

# %% [markdown]
# # XAI for RoBERTa+

# %% [markdown]
# ## 1.1 Attention Visualization using BertViz

# %%
from transformers import RobertaTokenizer, RobertaForSequenceClassification, utils
from bertviz import head_view
from IPython.display import display  # Correct import for display

# Suppress warnings
utils.logging.set_verbosity_error()

# Load your custom model and tokenizer
modelPATH = "/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier"
tokenizer = RobertaTokenizer.from_pretrained(modelPATH)
model = RobertaForSequenceClassification.from_pretrained(modelPATH, output_attentions=True)

# Input text
text = "Try Pelosi for treasonous actions against America and hang her"
# Example usage
# text = "She should be hanging from a rope at Guantanamo Bay!"
# target = 2  # Your target label

# text = "Go figure they are criminals lock them away for a very long time."
# target = 1  # Your target label

# text = "The whole biden administration is being ran like the mob."
# target = 0  # Your target label

inputs = tokenizer.encode(text, return_tensors='pt', add_special_tokens=True)

# Get model outputs including attentions
outputs = model(inputs)

# Extract attention weights (last element in outputs)
attention = outputs[-1]  # Output includes attention weights when output_attentions=True

# Convert input IDs to tokens
tokens = tokenizer.convert_ids_to_tokens(inputs[0])

# Visualize attention for all layers using head_view
head_view_output = head_view(attention, tokens)

# Display the visualization (interactive output)
# display(head_view_output)

# # Save the interactive attention visualization to an HTML file
# with open(modelPATH+'/bertviz_headview.html', 'w') as f:
#     f.write(head_view_output)

# %%
head_view(attention, tokens)

# %%
from transformers import RobertaTokenizer, RobertaForSequenceClassification, utils
from bertviz import model_view
from IPython.display import display  # Correct import for display

# Suppress warnings
utils.logging.set_verbosity_error()

# Load your custom model and tokenizer
modelPATH = "/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier"
tokenizer = RobertaTokenizer.from_pretrained(modelPATH)
model = RobertaForSequenceClassification.from_pretrained(modelPATH, output_attentions=True)

# Input text
# text = "Try Pelosi for treasonous actions against America and hang her"
# Example usage
# text = "She should be hanging from a rope at Guantanamo Bay!"
# target = 2  # Your target label

text = "Go figure they are criminals lock them away for a very long time."
target = 1  # Your target label

# text = "The whole biden administration is being ran like the mob."
# target = 0  # Your target label

inputs = tokenizer.encode(text, return_tensors='pt', add_special_tokens=True)

# Get model outputs including attentions
outputs = model(inputs)

# Extract attention weights (last element in outputs)
attention = outputs.attentions  # Corrected to use outputs.attentions for attention data

# Convert input IDs to tokens
tokens = tokenizer.convert_ids_to_tokens(inputs[0])

# Visualize attention for all layers using model_view
model_view_output = model_view(attention, tokens)

# Display the visualization (interactive output)
display(model_view_output)


# %%
from bertviz.transformers_neuron_view import RobertaForSequenceClassification, RobertaTokenizer
from bertviz.neuron_view import show

# Load your custom RoBERTa model and tokenizer
model_type = 'roberta'
model_version = "/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier"
model = RobertaForSequenceClassification.from_pretrained(model_version, output_attentions=True, num_labels=3)  # Ensure num_labels=3

# Specify the tokenizer files manually
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
# tokenizer = RobertaTokenizer.from_pretrained(model_version, 
#                                              vocab_file=model_version + "/vocab.json", merges_file=model_version + "/merges.txt")


# Example usage
# text = "She should be hanging from a rope at Guantanamo Bay!"
# target = 2  # Your target label

# text = "Go figure they are criminals lock them away for a very long time."
# target = 1  # Your target label

text = "The whole biden administration is being ran like the mob."
# target = 0  # Your target label

# Visualize neuron view for a specific layer and head
show(model, model_type, tokenizer, sentence_a=text, sentence_b=None,  display_mode="light", layer=23, head=15)

# %% [markdown]
# ## 2.1 Post-hoc using Captum

# %%
# !pip install captum
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
import numpy as np

# Load your custom tokenizer and model
modelPATH = "/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier"
tokenizer = RobertaTokenizer.from_pretrained(modelPATH)
model = RobertaForSequenceClassification.from_pretrained(modelPATH, num_labels=3)
model.eval()

# Example usage
# text = "She should be hanging from a rope at Guantanamo Bay!"
# target = 2  # Your target label

text = "Go figure they are criminals lock them away for a very long time."
target = 1  # Your target label

# text = "The whole biden administration is being ran like the mob."
# target = 0  # Your target label

# Tokenize the input
encoded = tokenizer(text, return_tensors="pt")

def predict(inputs, attention_mask=None):
    return model(inputs, attention_mask=attention_mask).logits

# Get model predictions
predictions = predict(encoded['input_ids'], encoded['attention_mask'])
predicted_label = predictions.argmax().item()

# Initialize Layer Integrated Gradients
lig = LayerIntegratedGradients(predict, model.roberta.embeddings)

# Calculate attributions
attributions_start, delta_start = lig.attribute(
    inputs=encoded['input_ids'],
    target=torch.tensor([target]),  # Using the specified target
    additional_forward_args=encoded['attention_mask'],
    return_convergence_delta=True
)

# Convert attributions to numpy for visualization
attributions_start = attributions_start.sum(dim=-1).squeeze().detach().numpy()  # Sum over embedding dimension
words = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

# Reconstruct original words and handle special tokens
original_words = [token if not token.startswith('Ġ') else token[1:] for token in words]  # Remove leading 'Ġ' from tokens

# Normalize attributions for better visualization (scaling between -1 and 1)
attributions_start = attributions_start / np.linalg.norm(attributions_start)

# Define class names
class_names = ["No Threat", "Judicial Threat", "Non-Judicial Threat"]

# Prepare data for visualization
result = viz.VisualizationDataRecord(
    attributions_start,
    predictions[0][predicted_label].item(),  # Predicted probability for the class
    class_names[predicted_label],  # Predicted label
    class_names[target],  # True label
    class_names[predicted_label],  # Attribution label
    attributions_start.sum(),  # Total attribution
    original_words,  # Use original words here
    delta_start  # Delta
)

# Visualize the results with colors
visualization_results = viz.visualize_text([result])

# %%
# !pip install captum
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
import numpy as np

# Load your custom tokenizer and model
modelPATH = "/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier"
tokenizer = RobertaTokenizer.from_pretrained(modelPATH)
model = RobertaForSequenceClassification.from_pretrained(modelPATH, num_labels=3)
model.eval()

# Example usage
# text = "She should be hanging from a rope at Guantanamo Bay!"
# target = 2  # Your target label

text = "Go figure they are criminals shoot them away for a very long time."
target = 1  # Your target label

# text = "The whole biden administration is being ran like the mob."
# target = 0  # Your target label

# Tokenize the input
encoded = tokenizer(text, return_tensors="pt")

def predict(inputs, attention_mask=None):
    return model(inputs, attention_mask=attention_mask).logits

# Get model predictions
predictions = predict(encoded['input_ids'], encoded['attention_mask'])
predicted_label = predictions.argmax().item()

# Initialize Layer Integrated Gradients
lig = LayerIntegratedGradients(predict, model.roberta.embeddings)

# Calculate attributions
attributions_start, delta_start = lig.attribute(
    inputs=encoded['input_ids'],
    target=torch.tensor([target]),  # Using the specified target
    additional_forward_args=encoded['attention_mask'],
    return_convergence_delta=True
)

# Convert attributions to numpy for visualization
attributions_start = attributions_start.sum(dim=-1).squeeze().detach().numpy()  # Sum over embedding dimension
words = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

# Reconstruct original words and handle special tokens
original_words = [token if not token.startswith('Ġ') else token[1:] for token in words]  # Remove leading 'Ġ' from tokens

# Normalize attributions for better visualization (scaling between -1 and 1)
attributions_start = attributions_start / np.linalg.norm(attributions_start)

# Define class names
class_names = ["No Threat", "Perturbed - Judicial Threat", "Non-Judicial Threat"]

# Prepare data for visualization
result = viz.VisualizationDataRecord(
    attributions_start,
    predictions[0][predicted_label].item(),  # Predicted probability for the class
    class_names[predicted_label],  # Predicted label
    class_names[target],  # True label
    class_names[predicted_label],  # Attribution label
    attributions_start.sum(),  # Total attribution
    original_words,  # Use original words here
    delta_start  # Delta
)

# Visualize the results with colors
visualization_results = viz.visualize_text([result])

# %%
# !pip install captum
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
import numpy as np

# Load your custom tokenizer and model
modelPATH = "/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier"
tokenizer = RobertaTokenizer.from_pretrained(modelPATH)
model = RobertaForSequenceClassification.from_pretrained(modelPATH, num_labels=3)
model.eval()

# Example usage
# text = "She should be hanging from a rope at Guantanamo Bay!"
# target = 2  # Your target label

text = "Go figure they are mob lock them away for a very long time."
target = 1  # Your target label

# text = "The whole biden administration is being ran like the mob."
# target = 0  # Your target label

# Tokenize the input
encoded = tokenizer(text, return_tensors="pt")

def predict(inputs, attention_mask=None):
    return model(inputs, attention_mask=attention_mask).logits

# Get model predictions
predictions = predict(encoded['input_ids'], encoded['attention_mask'])
predicted_label = predictions.argmax().item()

# Initialize Layer Integrated Gradients
lig = LayerIntegratedGradients(predict, model.roberta.embeddings)

# Calculate attributions
attributions_start, delta_start = lig.attribute(
    inputs=encoded['input_ids'],
    target=torch.tensor([target]),  # Using the specified target
    additional_forward_args=encoded['attention_mask'],
    return_convergence_delta=True
)

# Convert attributions to numpy for visualization
attributions_start = attributions_start.sum(dim=-1).squeeze().detach().numpy()  # Sum over embedding dimension
words = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

# Reconstruct original words and handle special tokens
original_words = [token if not token.startswith('Ġ') else token[1:] for token in words]  # Remove leading 'Ġ' from tokens

# Normalize attributions for better visualization (scaling between -1 and 1)
attributions_start = attributions_start / np.linalg.norm(attributions_start)

# Define class names
class_names = ["No Threat", "Perturbed - Judicial Threat", "Non-Judicial Threat"]

# Prepare data for visualization
result = viz.VisualizationDataRecord(
    attributions_start,
    predictions[0][predicted_label].item(),  # Predicted probability for the class
    "Judicial Threat",  # Predicted label
    class_names[target],  # True label
    class_names[predicted_label],  # Attribution label
    attributions_start.sum(),  # Total attribution
    original_words,  # Use original words here
    delta_start  # Delta
)

# Visualize the results with colors
visualization_results = viz.visualize_text([result])

# %%
# !pip install captum
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
import numpy as np

# Load your custom tokenizer and model
modelPATH = "/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier"
tokenizer = RobertaTokenizer.from_pretrained(modelPATH)
model = RobertaForSequenceClassification.from_pretrained(modelPATH, num_labels=3)
model.eval()

# Example usage
text = "Hope that teen spends the rest of their miserable existence in prison. POS"
target = 0  # Your target label

# Tokenize the input
encoded = tokenizer(text, return_tensors="pt")

def predict(inputs, attention_mask=None):
    return model(inputs, attention_mask=attention_mask).logits

# Get model predictions
predictions = predict(encoded['input_ids'], encoded['attention_mask'])
predicted_label = predictions.argmax().item()

# Initialize Layer Integrated Gradients
lig = LayerIntegratedGradients(predict, model.roberta.embeddings)

# Calculate attributions
attributions_start, delta_start = lig.attribute(
    inputs=encoded['input_ids'],
    target=torch.tensor([target]),  # Using the specified target
    additional_forward_args=encoded['attention_mask'],
    return_convergence_delta=True
)

# Convert attributions to numpy for visualization
attributions_start = attributions_start.sum(dim=-1).squeeze().detach().numpy()  # Sum over embedding dimension
words = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

# Reconstruct original words and handle special tokens
original_words = [token if not token.startswith('Ġ') else token[1:] for token in words]  # Remove leading 'Ġ' from tokens

# Normalize attributions for better visualization (scaling between -1 and 1)
attributions_start = attributions_start / np.linalg.norm(attributions_start)

# Define class names
class_names = ["No Threat", "Judicial Threat", "Non-Judicial Threat"]

# Prepare data for visualization
result = viz.VisualizationDataRecord(
    attributions_start,
    predictions[0][predicted_label].item(),  # Predicted probability for the class
    class_names[predicted_label],  # Predicted label
    class_names[target],  # True label
    class_names[predicted_label],  # Attribution label
    attributions_start.sum(),  # Total attribution
    original_words,  # Use original words here
    delta_start  # Delta
)

# Visualize the results with colors
visualization_results = viz.visualize_text([result])

# Get the generated HTML output and replace "True Label" with "Noisy Label"
html_output = visualization_results.data  # Captum stores HTML output in the data attribute
modified_html_output = html_output.replace("True Label", "Noisy Label")

# Render the modified output (in Jupyter, for example)
from IPython.core.display import display, HTML
display(HTML(modified_html_output))

# %%
# !pip install captum
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
import numpy as np

# Load your custom tokenizer and model
modelPATH = "/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier"
tokenizer = RobertaTokenizer.from_pretrained(modelPATH)
model = RobertaForSequenceClassification.from_pretrained(modelPATH, num_labels=3)
model.eval()

# Example usage
text = "ARREST HIM AND PUT A MASK ON HIM IN SOLITARY AND INJECT HIM WITH 5 DOSES"
target = 1  # Your target label

# Tokenize the input
encoded = tokenizer(text, return_tensors="pt")

def predict(inputs, attention_mask=None):
    return model(inputs, attention_mask=attention_mask).logits

# Get model predictions
predictions = predict(encoded['input_ids'], encoded['attention_mask'])
predicted_label = predictions.argmax().item()

# Initialize Layer Integrated Gradients
lig = LayerIntegratedGradients(predict, model.roberta.embeddings)

# Calculate attributions
attributions_start, delta_start = lig.attribute(
    inputs=encoded['input_ids'],
    target=torch.tensor([target]),  # Using the specified target
    additional_forward_args=encoded['attention_mask'],
    return_convergence_delta=True
)

# Convert attributions to numpy for visualization
attributions_start = attributions_start.sum(dim=-1).squeeze().detach().numpy()  # Sum over embedding dimension
words = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

# Reconstruct original words and handle special tokens
original_words = [token if not token.startswith('Ġ') else token[1:] for token in words]  # Remove leading 'Ġ' from tokens

# Normalize attributions for better visualization (scaling between -1 and 1)
attributions_start = attributions_start / np.linalg.norm(attributions_start)

# Define class names
class_names = ["No Threat", "Judicial Threat", "Non-Judicial Threat"]

# Prepare data for visualization
result = viz.VisualizationDataRecord(
    attributions_start,
    predictions[0][predicted_label].item(),  # Predicted probability for the class
    class_names[predicted_label],  # Predicted label
    class_names[target],  # True label
    class_names[predicted_label],  # Attribution label
    attributions_start.sum(),  # Total attribution
    original_words,  # Use original words here
    delta_start  # Delta
)

# Visualize the results with colors
visualization_results = viz.visualize_text([result])

# Get the generated HTML output and replace "True Label" with "Noisy Label"
html_output = visualization_results.data  # Captum stores HTML output in the data attribute
modified_html_output = html_output.replace("True Label", "Noisy Label")

# Render the modified output (in Jupyter, for example)
from IPython.core.display import display, HTML
display(HTML(modified_html_output))

# %% [markdown]
# ## 2.1.1 Attention Visualization using transformers-interpret (based on Captum)

# %%
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer

# Load your custom tokenizer and model
modelPATH = "/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier"
tokenizer = RobertaTokenizer.from_pretrained(modelPATH)
model = RobertaForSequenceClassification.from_pretrained(modelPATH, num_labels=3)
model.eval()

# Example usage
# text = "She should be hanging from a rope at Guantanamo Bay!"
# target = 2  # Your target label

text = "Go figure they are criminals lock them away for a very long time."
target = 1  # Your target label

# text = "The whole biden administration is being ran like the mob."
# target = 0  # Your target label

# Initialize the explainer
cls_explainer = SequenceClassificationExplainer(model, tokenizer)

# Get word attributions
word_attributions = cls_explainer(text)

# Set custom class names
cls_explainer.class_names = ["No Threat", "Judicial Threat", "Non-Judicial Threat"]

# Print predicted class
print("Predicted class:", cls_explainer.predicted_class_name)

# Visualize the results
cls_explainer.visualize()

# %%
# Define a custom function that generates the word_attributions
# and returns the visualization
def interpret_sentence(text):
  word_attributions = cls_explainer(text)
  return cls_explainer.visualize()

# %%
# Here we iterate over the random_sentences and return the 
# visualizations for each sentence
interpret_sentence(text=text)

# %%
word_attributions

# %% [markdown]
# ## 2.2 Post-hoc using LIME

# %%
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np
import torch

# Load your custom tokenizer and model
modelPATH = "/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier"
tokenizer = RobertaTokenizer.from_pretrained(modelPATH)
model = RobertaForSequenceClassification.from_pretrained(modelPATH, num_labels=3)
model.eval()

# Example usage
# text = "She should be hanging from a rope at Guantanamo Bay!"
# target = 2  # Your target label

text = "Go figure they are criminals lock them away for a very long time."
target = 1  # Your target label

# text = "The whole biden administration is being ran like the mob."
# target = 0  # Your target label

def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

# Initialize LIME explainer
explainer = LimeTextExplainer(class_names=["No Threat", "Judicial Threat", "Non-Judicial Threat"])

# Generate explanation for the input text
explanation = explainer.explain_instance(text, predict_proba, labels=[target])
# explanation = explainer.explain_instance(text, predict_proba, num_features=10, labels=[target])

# Display the explanation for the target class
explanation.show_in_notebook()
# explanation.show_in_notebook(text=True)

# explanation.save_to_file(modelPATH+'/lime_explanation.html')

# %%
import matplotlib.pyplot as plt

# Set the default font size to 8
plt.rcParams.update({'font.size': 8})

# Assuming 'explanation' is the result of your LIME explanation
with plt.style.context("ggplot"):
    fig = explanation.as_pyplot_figure()  # This generates the figure
    fig.set_size_inches(3.5, 2.5)  # Set the width to 3.5 inches and height to 2.5 inches

    # Remove the title
    ax = fig.gca()  # Get the current axes
    ax.set_title('')  # Set an empty string as the title to remove it

    # Save the figure as a PNG with 300 DPI at the specified modelPATH
    output_path = f"{modelPATH}/lime_local_explanation_plot.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')  # Save with 300 DPI and tight bounding box

    # Display the plot (optional)
    plt.show()  # Display the resized figure

# %%
explanation

# %% [markdown]
# ## 2.3 Post-hoc using SHAP

# %%
import pandas as pd
import transformers
import shap
from datasets import load_dataset

# Load your custom model and tokenizer
modelPATH = "/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier"
tokenizer = transformers.RobertaTokenizer.from_pretrained(modelPATH, use_fast=True)
model = transformers.RobertaForSequenceClassification.from_pretrained(modelPATH, num_labels=3).cuda()

# Custom data inputs
data = pd.DataFrame({
    "text": [
        "She should be hanging from a rope at Guantanamo Bay!",
        "Go figure they are criminals lock them away for a very long time.",
        "The whole biden administration is being ran like the mob."
    ],
    "target": [2, 1, 0]  # Target labels: 2 (Non-Judicial Threat), 1 (Judicial Threat), 0 (No Threat)
})

# Build a pipeline object to do predictions
pred = transformers.pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0,  # Use GPU
    return_all_scores=True,  # Get scores for all classes (0, 1, 2)
)

# Create a SHAP explainer for the pipeline
explainer = shap.Explainer(pred, output_names=["No Threat", "Judicial Threat", "Non-Judicial Threat"])

# Compute SHAP values for the 3 custom text samples
shap_values = explainer(data["text"][:3])

# Function to clean tokens by removing 'Ġ' prefix
def clean_shap_tokens(shap_values):
    cleaned_tokens = []
    for sample in shap_values.data:
        # Clean each token by removing the 'Ġ' prefix
        cleaned_sample = [token.replace("Ġ", "") for token in sample]
        cleaned_tokens.append(cleaned_sample)
    return cleaned_tokens

# Clean the SHAP tokens and replace in shap_values.data
shap_values.data = clean_shap_tokens(shap_values)

# Visualize the SHAP values for the target classes without the 'Ġ' prefix
shap.plots.text(shap_values)

# %%
shap_values[0,:,2]

# %%
# Assuming we're interested in the first sample and class 0 (No Threat)
shap_values_data = shap_values[0, :, 2]

# Separate positive and negative SHAP values for coloring
colors = ['red' if value > 0 else 'blue' for value in shap_values_data.values]

# Create a vertical bar plot using Matplotlib
plt.figure(figsize=(3.5, 4))
plt.barh(shap_values_data.data, shap_values_data.values, color=colors)  # Horizontal bar plot
plt.axvline(0, color='black', linewidth=0.8)  # Add a vertical line at x=0
plt.title("Mean SHAP Values for Class 'No Threat'", fontsize=8)
plt.xlabel("Mean SHAP Value", fontsize=8)
plt.ylabel("Tokens", fontsize=8)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
# Reduce the font size of y-ticks (tokens)
plt.xticks(fontsize=6)  
plt.yticks(fontsize=6) 
# Set x-axis limits
# plt.xlim(-0.05, 0.5)  # Adjust limits as needed
# plt.grid()
# Remove top and right borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming shap_values_data contains the SHAP values for the first sample
shap_values_data = shap_values[0, :, 2]

# Separate positive and negative SHAP values for coloring
colors = ['red' if value > 0 else 'blue' for value in shap_values_data.values]

# Create a scatter plot to mimic the beeswarm effect
plt.figure(figsize=(3.5, 4))
y_values = np.random.normal(0, 0.05, size=len(shap_values_data.values))  # Add some jitter for beeswarm effect
plt.scatter(shap_values_data.values, y_values, color=colors, alpha=0.6)  # Scatter plot
plt.axvline(0, color='black', linewidth=0.8)  # Add a vertical line at x=0
plt.title("SHAP Values for Class 'No Threat'", fontsize=8)
plt.xlabel("SHAP Value", fontsize=8)
plt.yticks([])  # Hide y-ticks since we don't need them in a beeswarm plot
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Reduce the font size of x-ticks
plt.xticks(fontsize=6)

# Remove top and right borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

# %%



