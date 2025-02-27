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

# ---------------------------
# Predict Button
# ---------------------------
if st.button("Predict"):
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

    # Use the model's predicted label as target
    target = predicted_label  
    st.write("### Predicted Label:", class_names[predicted_label])

    # Create three tabs for the XAI methods
    tabs = st.tabs(["Captum: Integrated Gradients Explanation", "LIME: Local Interpretable Explanation", "BertViz: Aggregated Attention Explanation"])

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
            inputs=encoded_captum['input_ids'],
            target=torch.tensor([target]),
            additional_forward_args=encoded_captum['attention_mask'],
            return_convergence_delta=True
        )
        attributions = attributions.sum(dim=-1).squeeze().detach().numpy()
        attributions = attributions / np.linalg.norm(attributions)
        words = tokenizer.convert_ids_to_tokens(encoded_captum['input_ids'][0])
        original_words = [w[1:] if w.startswith("Ġ") else w for w in words]
        return predictions, attributions, original_words, delta

    with tabs[0]:
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
        return explanation.as_html()

    with tabs[1]:
        st.header("LIME: Feature Importance")
        with st.spinner("Computing LIME explanation..."):
            lime_html = compute_lime_html(text, target)
            st.components.v1.html(lime_html, height=600, scrolling=True)


    # ======================================
    # BertViz Tab: Interactive Head View
    # ======================================
    with tabs[2]:
        st.header("BertViz Alternative: Token Importance")
        with st.spinner("Computing token importance from aggregated attention..."):
            # Use the final layer's attention: shape [num_heads, seq_len, seq_len]
            final_layer_attn = outputs.attentions[-1][0]
            # Average over heads to get a [seq_len, seq_len] matrix.
            avg_attn = final_layer_attn.mean(dim=0)
            # Compute a token importance score.
            # One option is to sum the attention each token receives from all other tokens.
            # (i.e., sum over columns) or average over rows.
            token_importance = avg_attn.sum(dim=0).cpu().detach().numpy()  # shape: (seq_len,)
            
            # Clean tokens: remove RoBERTa's "Ġ" prefix.
            tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
            tokens_clean = [t[1:] if t.startswith("Ġ") else t for t in tokens]
            
            # Create a horizontal bar chart.
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(range(len(tokens_clean)), token_importance, color='skyblue')
            ax.set_yticks(range(len(tokens_clean)))
            ax.set_yticklabels(tokens_clean, fontsize=12)
            ax.invert_yaxis()  # highest importance at the top
            ax.set_xlabel("Aggregated Attention Score", fontsize=12)
            ax.set_title("Token Importance from Aggregated Attention", fontsize=14)
            st.pyplot(fig)