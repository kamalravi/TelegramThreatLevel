import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from bertviz import head_view
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from lime.lime_text import LimeTextExplainer
import transformers
import time
import pandas as pd

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Total Freedom Interface",
    layout="wide"
)

# Dark mode toggle
dark_mode = st.sidebar.toggle("🌙 Enable Dark Mode", value=False)

# Inject Google Fonts and dynamic CSS
font_link = """
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
"""
st.markdown(font_link, unsafe_allow_html=True)

# Define dynamic style based on dark mode toggle
custom_style = f"""
<style>
body {{
    background-color: {"#121212" if dark_mode else "#f0f2f5"};
    font-family: 'Poppins', sans-serif;
}}

div[data-testid="stAppViewContainer"] > .main {{
    background-color: {"#1e1e1e" if dark_mode else "#ffffff"};
    border: 2px solid {"#333" if dark_mode else "#dee2e6"};
    border-radius: 15px;
    padding: 30px;
    margin: 30px auto;
    box-shadow: 0 0 15px rgba(0,0,0,0.1);
    max-width: 1100px;
}}

.stButton>button {{
    background-color: {"#bb86fc" if dark_mode else "#4a90e2"};
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 18px;
    font-size: 16px;
    font-weight: bold;
}}
.stButton>button:hover {{
    background-color: {"#9d6cfb" if dark_mode else "#3b7dd8"};
}}

.stSelectbox, .stRadio, .stCheckbox {{
    background-color: {"#2a2a2a" if dark_mode else "#ffffff"};
    padding: 10px;
    border-radius: 8px;
    box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
}}

h1, h2, h3 {{
    color: {"#e0e0e0" if dark_mode else "#2c3e50"};
    font-weight: 600;
}}

.stAlert.success {{
    background-color: {"#294436" if dark_mode else "#d1f0db"};
    color: {"#c2fbd7" if dark_mode else "#1a5d38"};
}}

::-webkit-scrollbar {{
    width: 8px;
}}
::-webkit-scrollbar-thumb {{
    background: #888;
    border-radius: 4px;
}}
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

st.title("Total Freedom Interface")

st.markdown("---")

@st.cache_resource(show_spinner=False)
def load_model():
    modelPATH = "/home/ka075671/raviProject/DataModelsResults/Results/PreTrainAgain_FineTune_RoBERTa_400/preTrainedModel/CustomPreTrainedClassifier"
    tokenizer = RobertaTokenizer.from_pretrained(modelPATH)
    model = RobertaForSequenceClassification.from_pretrained(
        modelPATH, num_labels=3, output_attentions=True
    )
    model.eval()
    class_names = ["No Threat", "Judicial Threat", "Non-Judicial Threat"]
    return model, tokenizer, class_names, modelPATH

model, tokenizer, class_names, modelPATH = load_model()

# ---------------------------
# Session state init
# ---------------------------
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "timings" not in st.session_state:
    st.session_state.timings = []


# ---------------------------
# Input Dropdown
# ---------------------------
sample_messages = [
    "Sample: Go figure they are criminals lock them away for a very long time.",
    "No Shirt No Shoes No Service",
    "Sending you and your family many thoughts and prayers. God bless!",
    "So then can we now sue them for all the  stress they’ve given",
    "They need to be slapped with the constitution and thrown in jail",
    "“Wood Chipper” Death Penalty for everyone involved.",
    "If the Military doesn't do something I may not be able to control myself."
]


if st.button("Start Timer"):
    st.session_state.start_time = time.time()
    st.success("Timer started. Now begin your task.")

text = st.selectbox("Select a message for explanation:", sample_messages, index=0)
st.session_state.selected_text = text


st.markdown("---")


# ---------------------------
# Prediction Trigger and State
# ---------------------------
if "predicted" not in st.session_state:
    st.session_state.predicted = False

if st.button("Predict") or st.session_state.predicted:
    st.session_state.predicted = True
    st.write("### Input Text")
    st.write(text)

    @st.cache_data(show_spinner=False)
    def get_prediction(text):
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        outputs = model(**encoded)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        return outputs, predicted_label, encoded

    with st.spinner("Computing model prediction..."):
        outputs, predicted_label, encoded = get_prediction(text)

    target = predicted_label
    st.write("### Predicted Label:", class_names[predicted_label])

    show_xai = st.checkbox("Show XAI Options")
    st.session_state.show_xai = show_xai

    if show_xai:
        xai_option = st.radio(
            "Select an explanation method:",
            ["XAI Method 1", "XAI Method 2", "XAI Method 3"],
            index=None,
            key="xai_choice"
        )

        if xai_option == "XAI Method 1":  # Captum: Integrated Gradients
            st.header("XAI Method 1")
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

            with st.spinner("Computing explanation..."):
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
                st.components.v1.html(captum_html, height=150, scrolling=True)

        elif xai_option == "XAI Method 2":
            st.header("XAI Method 2")
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

            with st.spinner("Computing explanation..."):
                lime_html = compute_lime_html(text, target)
                st.components.v1.html(lime_html, height=300, scrolling=True)

        elif xai_option == "XAI Method 3":
            st.header("XAI Method 3")
            with st.spinner("Computing explanation..."):
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

        # End timer and save duration
        if st.button("End Timer"):
            end_time = time.time()
            duration = round(end_time - st.session_state.start_time, 2)
            st.session_state.timings.append({
                "Message": st.session_state.selected_text,
                "Duration (s)": duration
            })
            st.success(f"Time recorded for: {st.session_state.selected_text} - {duration} seconds")
            
            # Reset session state except timings
            for key in ["predicted", "xai_choice", "show_xai", "start_time"]:
                if key in st.session_state:
                    del st.session_state[key]
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()


st.markdown("---")

# Option to export all timing data
if st.button("Export Timing CSV"):
    fname = "/home/ka075671/raviProject/CODE/XAI/timings_output.csv"
    df = pd.DataFrame(st.session_state.timings)
    df.to_csv(fname, index=False)
    st.success(f"Timing data exported to {fname}")
