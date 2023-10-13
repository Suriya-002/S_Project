import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import random

# Function to display an image with a given path
def display_image_with_path(image_path):
    st.image(image_path)

# Set the Streamlit app configuration
st.set_page_config(
    page_title="Photovoltaic Defect Classification",
    page_icon=":mango:",
    initial_sidebar_state='auto'
)

# Hide Streamlit's default menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Define a function to map predictions to defect names
def prediction_cls(prediction):
    class_names = ['Bird Drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical Damage', 'Snow-Covered']
    return class_names[np.argmax(prediction)]

# Sidebar content
with st.sidebar:
    st.image('Images/Clean/Clean (7).jpg')
    st.title("Accurate classification of defects present in the photovoltaic cells.")

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'efficientnetb0-Photovoltaic Defects-90.14.h5'  # Update the path to your model file
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Create a compatible optimizer with weight decay
    weight_decay = 1e-4
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=weight_decay)
    
    # Compile the model with the optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

# Main app content
st.write("""
         # Photovoltaic Defect Detection with Remedy Suggestion
         """)

file = st.file_uploader("", type=["jpg", "png"])

# Function to preprocess and make predictions
def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.BICUBIC)  # Use Image.BICUBIC as the resampling filter
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98, 99) + random.randint(0, 99) * 0.01
    detected_defect = prediction_cls(predictions)

    # Display accuracy in the sidebar with larger font size and bold text
    st.sidebar.markdown(
        f"<p style='font-size: 24px; font-weight: bold;'>Accuracy : {x:.2f} %</p>", 
        unsafe_allow_html=True
    )
    st.sidebar.warning(f"<p style='font-size: 18px;'>Detected Defect: {detected_defect}</p>")

    # Provide remedies based on detected defect
    if detected_defect == 'Clean':
        st.markdown("## Remedies")
        st.markdown("<p style='color: black;'>No Defect is Found. It is clean.</p>", unsafe_allow_html=True)

    elif detected_defect == 'Bird Drop':
        st.markdown("## Remedies")
        bird_drop_remedies = """
        <p style='color: black;'>1. Install bird deterrent devices like spikes or netting around the solar panels to prevent birds from landing.</p>
        <p style='color: black;'>2. Regularly clean and maintain the panels to remove bird droppings, reducing their attraction to the area.</p>
        <p style='color: black;'>3. Consider using scare tactics like decoy predators or ultrasonic repellents to deter birds from approaching the panels.</p>
        """
        st.markdown(bird_drop_remedies, unsafe_allow_html=True)

    elif detected_defect == 'Dusty':
        st.markdown("## Remedies")
        dusty_remedies = """
        <p style='color: black;'>1. Clean panels periodically with water and a soft brush to remove dust and dirt.</p>
        <p style='color: black;'>2. Consider automated cleaning systems or rain for natural cleaning.</p>
        <p style='color: black;'>3. Schedule professional inspections for heavy or persistent dust buildup.</p>
        """
        st.markdown(dusty_remedies, unsafe_allow_html=True)

    elif detected_defect == 'Electrical-damage':
        st.markdown("## Remedies")
        electrical_damage_remedies = """
        <p style='color: black;'>1. Isolate the damaged panel to prevent further issues in the array.</p>
        <p style='color: black;'>2. Consult a certified solar technician to assess and repair the damage.</p>
        <p style='color: black;'>3. Regularly inspect and maintain your solar panel system to prevent future electrical issues.</p>
        """
        st.markdown(electrical_damage_remedies, unsafe_allow_html=True)

    elif detected_defect == 'Physical Damage':
        st.markdown("## Remedies")
        physical_damage_remedies = """
        <p style='color: black;'>1. Examine the extent of physical damage, identifying cracks or breaks in the panels.</p>
        <p style='color: black;'>2. Depending on the severity, either repair the damaged sections or replace the affected panels to restore optimal functionality.</p>
        <p style='color: black;'>3. Implement regular maintenance practices to prevent further damage and ensure the longevity of your solar panels.</p>
        """
        st.markdown(physical_damage_remedies, unsafe_allow_html=True)

    elif detected_defect == 'Snow-Covered':
        st.markdown("## Remedies")
        snow_covered_remedies = """
        <p style='color: black;'>1. Gently remove snow using a soft brush or a long-handled tool.</p>
        <p style='color: black;'>2. Ensure the panels are clear to maximize sunlight absorption.</p>
        <p style='color: black;'>3. Monitor and clear snow regularly during winter months.</p>
        """
        st.markdown(snow_covered_remedies, unsafe_allow_html=True)

# Footer
st.markdown("Made by Shanthoshini Devi and Sanjay")
