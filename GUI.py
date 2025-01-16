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
                    - Accuracy represents the overall correctness of the model. It is the ratio of correctly predicted cases (both positive and negative) to the total number of cases.
                - **Precision**: {precision:.2%} 
                    - Precision measures the proportion of true positive predictions among all positive predictions. It answers the question: "Of all the images the model labeled as pneumonia, how many were correct?"
                - **Recall**: {recall:.2%} 
                    - Recall measures the proportion of true positive predictions among all actual positive cases. It answers the question: "Of all the actual pneumonia cases, how many did the model correctly identify?"
                - **F1 Score**: {f1:.2%} 
                    - The F1 Score is the harmonic mean of precision and recall, providing a balance between the two. It is useful when there is an uneven class distribution.
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
                st.markdown(
                    """
                    **Grad-CAM** highlights the regions of the image that are most important for the model's prediction.
                    Warmer colors (red, orange) indicate areas that strongly influenced the prediction, while cooler colors (blue) indicate less influential regions.
                    """
                )

            if os.path.exists(lime_save_path):
                st.subheader(f"Key regions that indicate the presence of {predicted_class}")
                st.image(lime_save_path, caption="LIME Visualization", use_container_width=True)
                st.markdown(
                    f"""
                    **LIME (Local Interpretable Model-agnostic Explanations)** provides insights into the areas of the image that contributed to the model's prediction.
                    It identifies specific segments or superpixels that were most significant in classifying the image as {predicted_class}.
                    """
                )
