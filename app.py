import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Model Prediction
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
def model_prediction(test_image):
    model = load_model('./bestmodel_resnet.h5')
    img = tf.keras.preprocessing.image.load_img(test_image,target_size=(200,200))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array=img_array/255 #scaling is important 
    img_array = tf.expand_dims(img_array,0)
    predictions = model.predict(img_array)
    predicted_class = labels[np.argmax(predictions[0])]
    confidence  = round(100*(np.max(predictions[0])),2)
    return predicted_class,confidence

#Sidebar
st.sidebar.title('Dashboard')
app_mode = st.sidebar.selectbox('Seelct Page', ['Home','About','Disease Recognition'])

#Home Page
if (app_mode == 'Home'):
    st.header('Brain Tumor Detection System')
    st.markdown("""
    Welcome to the Brain Tumor Detection System! 🌿🔍
    
    Brain tumor is the accumulation or mass growth of abnormal cells in the brain. There are basically two types of tumors, malignant and benign. Malignant tumors can be life-threatening based on the location and rate of growth. Hence timely intervention and accurate detection is of paramount importance when it comes to brain tumors. This project focusses on classifying 3 types of brain tumors based on its loaction from normal cases i.e no tumor 

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** The system will process the image using advanced algorithms to classify the tumors.
    3. **Results:** View the results and recommendations for further action.



    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of Brain Tumor Detection System!

    """)
    #About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                The dataset used for this model is taken from Brain Tumor MRI Dataset available on Kaggle.
                The distribution of images in training data are as follows:
	            •	Pituitary tumor (916)
                •	Meningioma tumor (906)
                •	Glioma tumor (900)
                •	No tumor (919) 

                The distribution of images in testing data are as follows:
                •	Pituitary tumor (200)
                •	Meningioma tumor (206)
                •	Glioma tumor (209)
                •	No tumor (278


                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Brain Tumor Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        with st.spinner("Please wait..."):
            st.write("Prediction")
            predicted_class,confidence = model_prediction(test_image)
            st.success("The model is predicting it's a {}".format(predicted_class))
            st.success("Confidence in prediction is {} %".format(confidence))
        