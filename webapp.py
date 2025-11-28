import streamlit as st
from tensorflow import zeros, int64
from GPT import GPT, get_batch, decode

st.title("GPT Text Generation")

@st.cache_resource
def load_model():
    """Loads and caches the GPT model."""
    print("--- Loading TRAINED model ---")
    model = GPT(training=False)
    # Run a forward pass to build the model before loading weights
    x_val_batch, y_val_batch = get_batch("val")
    _, _ = model(x_val_batch, y_val_batch)
    model.load_weights("gpt_model.weights.h5")
    print("--- Model loaded ---")
    return model

model = load_model()

max_tokens = st.slider("Select number of tokens to generate", 50, 1000, 250)

if st.button("Generate Text"):
    st.write("Generating text...")
    context_trained = zeros((1, 1), dtype=int64)

    placeholder = st.empty()
    generated_tokens = []
    for token in model.generate(idx=context_trained, max_new_tokens=max_tokens):
        generated_tokens.append(token.numpy().tolist()[0][0])
        decoded_text_trained = decode(generated_tokens)
        placeholder.markdown(f"**Generated Text:**\n\n---\n\n{decoded_text_trained}")