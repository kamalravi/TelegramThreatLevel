import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaForSequenceClassification, utils
from bertviz import head_view
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from lime.lime_text import LimeTextExplainer
import transformers

# ---------------------------
# 1. Load Model, Tokenizer & Configurations
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    modelPATH = "/home/ravi/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier"
    tokenizer = RobertaTokenizer.from_pretrained(modelPATH)
    model = RobertaForSequenceClassification.from_pretrained(
        modelPATH, num_labels=3, output_attentions=True
    )
    model.eval()
    class_names = ["No Threat", "Judicial Threat", "Non-Judicial Threat"]
    return model, tokenizer, class_names, modelPATH

model, tokenizer, class_names, modelPATH = load_model()

# ---------------------------
# 2. Define Input Text and Target Label
# ---------------------------
text = "Go figure they are criminals lock them away for a very long time."
target = 1  # Target label for explanation

st.title("XAI Explanation Dashboard")
st.write("### Input Text")
st.write(text)

# ---------------------------
# 3. Model Prediction (Shared)
# ---------------------------
@st.cache_data(show_spinner=False)
def get_prediction(text):
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    outputs = model(**encoded)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    return outputs, predicted_label, encoded

with st.spinner("Computing model prediction..."):
    outputs, predicted_label, encoded = get_prediction(text)
st.write("### Predicted Label:", class_names[predicted_label])

# Create three tabs for the three XAI methods
tabs = st.tabs(["BertViz", "Captum", "LIME"])

# ======================================
# BertViz Tab: Interactive Head View
# ======================================
with tabs[0]:
    st.header("BertViz: Interactive Attention")
    with st.spinner("Loading BertViz explanation..."):
        # Get attentions and tokens for visualization
        attentions = outputs.attentions  # Tuple: one tensor per layer
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

        # Attempt to get BertViz interactive head_view
        hv = head_view(attentions, tokens)

        if hv is not None and hasattr(hv, "_repr_html_"):
            hv_html = hv._repr_html_()  # Get HTML representation
            st.components.v1.html(hv_html, height=600, scrolling=True)
        else:
            st.write("Interactive BertViz visualization is not available in this environment. Displaying aggregated attention instead.")
            
            # Compute an aggregated attention heatmap fallback
            final_layer_attn = outputs.attentions[-1][0]  # shape: [num_heads, seq_len, seq_len]
            # Average only over heads to keep a 2D matrix:
            avg_attn = final_layer_attn.mean(dim=0).cpu().detach().numpy()  # shape: (seq_len, seq_len)

            # Clean tokens (remove RoBERTa's "Ġ" prefix)
            tokens_clean = [token[1:] if token.startswith("Ġ") else token for token in tokens]

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(avg_attn, cmap='viridis')
            ax.set_xticks(np.arange(len(tokens_clean)))
            ax.set_xticklabels(tokens_clean, rotation=90, fontsize=8)
            ax.set_yticks(np.arange(len(tokens_clean)))
            ax.set_yticklabels(tokens_clean, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)

# ======================================
# Captum Tab: Integrated Gradients Explanation
# ======================================
@st.cache_data(show_spinner=False)
def compute_captum(text, target):
    encoded_captum = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    def predict(inputs, attention_mask=None):
        return model(inputs, attention_mask=attention_mask).logits
    predictions = predict(encoded_captum['input_ids'], encoded_captum['attention_mask'])
    lig = LayerIntegratedGradients(predict, model.roberta.embeddings)
    attributions, delta = lig.attribute(
        inputs=encoded_captum['input_ids'],  # Fixed typo here
        target=torch.tensor([target]),
        additional_forward_args=encoded_captum['attention_mask'],
        return_convergence_delta=True
    )
    attributions = attributions.sum(dim=-1).squeeze().detach().numpy()
    attributions = attributions / np.linalg.norm(attributions)
    words = tokenizer.convert_ids_to_tokens(encoded_captum['input_ids'][0])
    original_words = [w[1:] if w.startswith("Ġ") else w for w in words]
    return predictions, attributions, original_words, delta


with tabs[1]:
    st.header("Captum: Integrated Gradients")
    with st.spinner("Computing Captum explanation..."):
        predictions, attributions, original_words, delta = compute_captum(text, target)
        # Prepare Captum visualization record
        result = viz.VisualizationDataRecord(
            attributions,
            predictions[0][predicted_label].item(),
            class_names[predicted_label],
            class_names[target],
            class_names[predicted_label],
            attributions.sum(),
            original_words,
            delta
        )
        captum_vis = viz.visualize_text([result])
        captum_html = captum_vis._repr_html_()
        st.components.v1.html(captum_html, height=400, scrolling=True)

# ======================================
# LIME Tab: Local Interpretable Explanations
# ======================================
@st.cache_data(show_spinner=False)
def compute_lime_html(text, target):
    def predict_proba(texts):
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    lime_explainer = LimeTextExplainer(class_names=class_names)
    explanation = lime_explainer.explain_instance(text, predict_proba, labels=[target])
    # Return the HTML representation of the LIME explanation
    return explanation.as_html()

with tabs[2]:
    st.header("LIME: Feature Importance")
    with st.spinner("Computing LIME explanation..."):
        lime_html = compute_lime_html(text, target)
        st.components.v1.html(lime_html, height=600, scrolling=True)

