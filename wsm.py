import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="iR5Xp9mLhRQFRQjeQWjm"
)

# Streamlit app
st.title("Women Safety Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Save the uploaded file temporarily
    with open("happy.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Perform inference
    result = CLIENT.infer("happy.jpg", model_id="women-safety-trnib/1")

    # Display the result
    st.write("Inference Result:")
    st.json(result)