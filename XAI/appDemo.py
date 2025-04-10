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
# Page Config and Custom CSS for Wider Layout and Bigger Text Box
# ---------------------------
st.set_page_config(
    page_title="USEnTEL: User Satisfaction and Experience in Threat Explainability Tool",
    layout="wide"
)
# ---------------------------
# Running Title
# ---------------------------
st.title("USEnTEL: User Satisfaction and Experience in Threat Explainability Tool")


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
# 2. Input Text Box with Bigger Font
# ---------------------------
default_text = "Go figure they are criminals lock them away for a very long time."
st.markdown('<div class="big-text-area">', unsafe_allow_html=True)
text = st.text_area("Enter a text for explanation:", default_text)
st.markdown('</div>', unsafe_allow_html=True)

# Predict when button is clicked
if st.button("Predict"):
    with st.spinner("Computing model prediction..."):
        outputs, predicted_label, encoded = get_prediction(text)
        target = predicted_label
        st.session_state['prediction_result'] = (outputs, predicted_label, encoded, target)

# Tabs appear only after prediction
if 'prediction_result' in st.session_state:
    outputs, predicted_label, encoded, target = st.session_state['prediction_result']
    st.write("### Predicted Label:", class_names[predicted_label])

    # Tab selector
    selected_tab = st.radio("Select Explanation Method", ["Captum", "LIME", "BertViz"], horizontal=True)

    if selected_tab == "Captum":
        st.header("Captum: Integrated Gradients")
        with st.spinner("Computing Captum explanation..."):
            predictions, attributions, original_words, delta = compute_captum(text, target)
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
            st.components.v1.html(captum_vis._repr_html_(), height=400, scrolling=True)
    
    elif selected_tab == "LIME":
        st.header("LIME: Feature Importance")
        with st.spinner("Computing LIME explanation..."):
            lime_html = compute_lime_html(text, target)
            st.components.v1.html(lime_html, height=600, scrolling=True)

    elif selected_tab == "BertViz":
        st.header("BertViz: Token Attention")
        with st.spinner("Computing token importance..."):
            final_layer_attn = outputs.attentions[-1][0]
            avg_attn = final_layer_attn.mean(dim=0)
            token_importance = avg_attn.sum(dim=0).cpu().detach().numpy()
            tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
            tokens_clean = [t[1:] if t.startswith("Ġ") else t for t in tokens]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(range(len(tokens_clean)), token_importance, color='skyblue')
            ax.set_yticks(range(len(tokens_clean)))
            ax.set_yticklabels(tokens_clean, fontsize=12)
            ax.invert_yaxis()
            ax.set_xlabel("Aggregated Attention Score", fontsize=12)
            ax.set_title("Token Importance from Aggregated Attention", fontsize=14)
            st.pyplot(fig)
