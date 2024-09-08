import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import datetime

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="iR5Xp9mLhRQFRQjeQWjm"
)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Custom CSS for glassmorphism effect and responsiveness
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .glassmorphism {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 20px;
        margin: 20px;
        text-align: center;
    }
    .result-box {
        padding: 10px;
        border-radius: 5px;
        width: fit-content;
        margin: auto;
        text-align: center;
        background-color: #444444;
    }
    .helpline-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 20px;
        margin: 20px;
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 10px;
        margin-top: 20px;
        font-size: 1em;
        color: #aaaaaa;
    }
    @media (max-width: 768px) {
        .glassmorphism {
            margin: 10px;
            padding: 15px;
        }
        .result-box {
            padding: 8px;
        }
        .helpline-box {
            margin: 10px;
            padding: 15px;
        }
    }
    </style>
    """, unsafe_allow_html=True
)

# Main content area
st.markdown(
    """
    <div class='glassmorphism'>
        <h1>Women Safety Detection AI</h1>
    </div>
    """, unsafe_allow_html=True
)

st.write("Upload Image or Video")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

def display_result(result_text):
    st.markdown(
        f"""
        <div class='result-box'>
            <h3>{result_text}</h3>
        </div>
        """, unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Initialize progress bar
    progress_bar = st.progress(0)

    # Save the uploaded file temporarily
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    progress_bar.progress(25)  # Update progress

    # Perform inference
    result = CLIENT.infer("uploaded_image.jpg", model_id="women-safety-trnib/1")
    progress_bar.progress(75)  # Update progress

    # Extract and display the result
    if result and "predictions" in result:
        safe_count = 0
        abuse_count = 0
        safe_confidences = []
        abuse_confidences = []

        for prediction in result["predictions"]:
            class_name = prediction["class"]
            confidence = prediction["confidence"]

            if class_name == "Safe":
                safe_count += 1
                safe_confidences.append(confidence)
            else:
                abuse_count += 1
                abuse_confidences.append(confidence)

        # Initial evaluation of conditions
        final_result = "Inconclusive"
        if abuse_count >= 2:
            final_result = "Abuse detected"
        if safe_count >= 3 or (safe_count >= 1 and abuse_count == 1):
            final_result = "Safe detected"

        # Re-evaluate to ensure accuracy
        if abuse_count >= 2:
            final_result = "Abuse detected"
        elif safe_count >= 3:
            final_result = "Safe detected"

        # Specific condition: 3 Safe and 1 Abuse with exactly 4 predictions
        if len(result["predictions"]) == 4 and safe_count == 3 and abuse_count == 1:
            final_result = "Abuse detected"

        # Specific condition: 2 Safe and 1 Abuse with exactly 3 predictions
        if len(result["predictions"]) == 3 and safe_count == 2 and abuse_count == 1:
            final_result = "Abuse detected"

        # Specific condition: More Safe than Abuse with exactly 8 predictions
        if len(result["predictions"]) == 8:
            if safe_count < abuse_count:
                final_result = "Safe detected"
            elif abuse_count < safe_count:
                final_result = "Abuse detected"
            else:
                final_result = "Inconclusive"

        # Specific condition: 4 predictions and all Abuse confidences are lower than Safe confidences
        if len(result["predictions"]) == 4 and all(ac < min(safe_confidences) for ac in abuse_confidences):
            final_result = "Safe detected"

        # Specific condition: 3 predictions and all are Safe but any Safe confidence is less than 0.50
        if len(result["predictions"]) == 3 and safe_count == 3 and any(sc < 0.50 for sc in safe_confidences):
            final_result = "Abuse detected"

        # Specific condition: 3 predictions and all are Abuse but any Abuse confidence is less than 0.50
        if len(result["predictions"]) == 3 and abuse_count == 3 and any(ac < 0.50 for ac in abuse_confidences):
            final_result = "Safe detected"

        # Display the result
        display_result(final_result)

        # Display helpline numbers if abuse is detected
        if final_result == "Abuse detected":
            st.markdown(
                """
                <div class='helpline-box'>
                    <h2>Helpline Numbers</h2>
                    <p>National Helpline: 1091</p>
                    <p>Women Helpline (All India): 181</p>
                    <p>Police: 100</p>
                    <p>Emergency Response Support System: 112</p>
                </div>
                """, unsafe_allow_html=True
            )

        # Save to history
        st.session_state.history.append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": uploaded_file.name,
            "result": final_result
        })
    else:
        st.write("No predictions found.")
    progress_bar.progress(100)  # Update progress to complete

# Footer
st.markdown(
    """
    <div class='footer'>
        Made for Women , Made in India
    </div>
    """, unsafe_allow_html=True
)