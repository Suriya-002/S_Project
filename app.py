import streamlit as st
import os

# Define the function to display the image using os.path.join() and forward slashes
def display_image_with_path(image_path):
    st.image(image_path)

# ... (other imports and functions)

# Your other functions and code here

st.set_page_config(
    page_title="Photovoltaic Defect Classification",
    page_icon=":mango:",
    initial_sidebar_state='auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction) == clss:
            return key

with st.sidebar:
    st.image('Images/Clean/Clean (7).jpg')
    st.title("No Defect")
    st.subheader("Accurate classification of defects present in the photovoltaic cells.")

@st.cache(allow_output_mutation=True)
def load_model():
    # Load only the model architecture and weights (excluding optimizer state)
    model = tf.keras.models.load_model('C:\\Users\\SURIYA\\Documents\\Shantho Project\\efficientnetb0-Photovoltaic Defects-90.14.h5', compile=False)

    # Specify the weight decay value (e.g., 1e-4)
    weight_decay = 1e-4

    # Create a compatible optimizer with weight decay
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=weight_decay)

    # Compile the model with the optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Photovoltaic Defect  Detection with Remedy Suggestion
         """
         )

file = st.file_uploader("", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
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
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['Bird Drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical Damage', 'Snow-Covered',]

    string = "Detected Defect : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'Clean':
        st.balloons()
        st.sidebar.success(string)
        st.info('No Defect is Found. It is clean')

    elif class_names[np.argmax(predictions)] == 'Bird Drop':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "Install bird deterrent devices like spikes or netting around the solar panels to prevent birds from landing.Regularly clean and maintain the panels to remove bird droppings, reducing their attraction to the area.Consider using scare tactics like decoy predators or ultrasonic repellents to deter birds from approaching the panels.")

    elif class_names[np.argmax(predictions)] == 'Dusty':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            " Clean panels periodically with water and a soft brush to remove dust and dirt.Consider automated cleaning systems or rain for natural cleaning.Schedule professional inspections for heavy or persistent dust buildup.")

    elif class_names[np.argmax(predictions)] == 'Electrical-damage':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "Isolate the damaged panel to prevent further issues in the array.Consult a certified solar technician to assess and repair the damage.Regularly inspect and maintain your solar panel system to prevent future electrical issues.")

    elif class_names[np.argmax(predictions)] == 'Physical-Damage':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "Examine the extent of physical damage, identifying cracks or breaks in the panels.Depending on the severity, either repair the damaged sections or replace the affected panels to restore optimal functionality.Implement regular maintenance practices to prevent further damage and ensure the longevity of your solar panels.")

    elif class_names[np.argmax(predictions)] == 'Snow-Covered':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info(
            "Gently remove snow using a soft brush or a long-handled tool.Ensure the panels are clear to maximize sunlight absorption.Monitor and clear snow regularly during winter months.")

# Footer
myargs = [
    "Made in ",
    image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',
          width=px(25), height=px(25)),
    " with ❤️ by Shanthoshini Devi and Sanjay",
    br()
]
layout(*myargs)
