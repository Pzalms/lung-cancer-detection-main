import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved MobileNet model
model = load_model("mobilenet_model.h5")

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize pixel values
    return img

# Function to make predictions
def predict(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    return prediction

# Sidebar content
st.sidebar.markdown("**About Lung Cancer Detection**")
st.sidebar.write("This app uses a CNN model to detect lung cancer from CT scan images.")
st.sidebar.write("Lung cancer is the uncontrolled growth of abnormal cells in one or both lungs. Early detection is crucial for effective treatment.")
st.sidebar.write("Symptoms of lung cancer include persistent cough, chest pain, hoarseness, weight loss, and shortness of breath.")
st.sidebar.write("Treatment options may include surgery, chemotherapy, radiation therapy, immunotherapy, or targeted drug therapy.")

# Streamlit app
def main():
    st.title("Lung Cancer Detection App")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("Predict"):
            # Make prediction
            prediction = predict(uploaded_file)

            # Display prediction
            if prediction[0][0] > 0.5:
                st.error("**Prediction: Non-Cancerous**")
            else:
                st.success("**Prediction: Cancerous**")

if __name__ == "__main__":
    main()
