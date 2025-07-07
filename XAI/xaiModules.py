from transformers import AutoTokenizer, AutoModel, utils
from bertviz import head_view
from IPython.display import display  # Correct import for display

utils.logging.set_verbosity_error()  # Suppress standard warnings

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased", output_attentions=True)

# Input text
inputs = tokenizer.encode("The cat sat on the mat", return_tensors='pt')
outputs = model(inputs)

# Extract attention weights
attention = outputs[-1]  # Output includes attention weights when output_attentions=True

# Convert input IDs to tokens
tokens = tokenizer.convert_ids_to_tokens(inputs[0])

# Use head_view to visualize attention and capture the output
head_view_output = head_view(attention, tokens)

# Display the visualization
display(head_view_output)  # This should now properly show the visualizations
