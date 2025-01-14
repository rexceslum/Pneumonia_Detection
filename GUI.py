import streamlit as st
from PIL import Image
import os
from GUI_trigger_point import predict, get_performance

# Define paths
model_save_path = "./model/pytorch_resnet50_model.pth"
gradcam_save_path = "./result/gradcam_result.png"
lime_save_path = "./result/lime_result.png"

# Set page layout to 'wide' to use the entire width of the page
st.set_page_config(layout="wide")

# Streamlit app
st.title("Pneumonia Detection with Explainable AI")

# Create a container to hold the entire content
with st.container():
    # Create a 2x2 grid layout using st.columns for sections
    col1, col2 = st.columns([1, 1])  # left column, right column

    # Section 1: File uploader and display uploaded image (Top-left)
    with col1:
        st.header("Upload Chest X-Ray Image")
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

        if uploaded_file is not None:
            # Display uploaded image
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Image", use_container_width=True)

            # Run prediction
            with st.spinner("Processing..."):
                predicted_class = predict(input_image)
                
        if st.button("Get Model Performance"):
            with st.spinner("Evaluating model performance..."):
                accuracy, precision, recall, f1 = get_performance()
            # Display performance metrics
            st.header("Model Performance Metrics")
            st.markdown(
                f"""
                - **Accuracy**: {accuracy:.2%}
                - **Precision**: {precision:.2%}
                - **Recall**: {recall:.2%}
                - **F1 Score**: {f1:.2%}
                """
            )

    # Section 2: Display predicted class (Top-right)
    with col2:
        if uploaded_file is not None:
            st.header(f"Predicted Class: **{predicted_class}**")
            if predicted_class == "PNEUMONIA":
                st.markdown(
                    """
                    **Pneumonia Detected!**
                    Pneumonia is an infection that inflames the air sacs in one or both lungs. It can cause mild to severe symptoms, including chest pain, coughing, and fever.
                    Immediate medical consultation is recommended.
                    """
                )
            else:
                st.markdown(
                    """
                    **Normal Chest X-Ray!**
                    The chest X-ray does not show signs of pneumonia. However, consult a healthcare professional if symptoms persist.
                    """
                )
            
            # Display Grad-CAM and LIME results
            if os.path.exists(gradcam_save_path):
                st.subheader("Features that contribute to the prediction")
                st.image(gradcam_save_path, caption="Grad-CAM Visualization", use_container_width=True)

            if os.path.exists(lime_save_path):
                st.subheader(f"Key regions that indicate the presence of {predicted_class}")
                st.image(lime_save_path, caption="LIME Visualization", use_container_width=True)
