# Image-Classifier

This is a simple image classifier(Classifies the input image as a Medical/NonMedical image). The model used for this approach is a Convolutional Neural Network.
The data used for training the CIFAR-10 and Hugging Face datasets.
**Links** 
1) [https://huggingface.co/datasets/tanzuhuggingface/brainmri](url)
2) [https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data](url)


# Setup

1) **Python 3.12.3**
2) install **requirements.txt**
3) Google Colab for training the model

# Workflow

image ----> preprocessing ---->  SMOTE ----> CNN Model[Training] ----> Prediction[for Evaluation]

1) In preprocessing, images are resized to 128*128*3, and normalization is applied to pixel values are within range(In addition you can also add denoising, contrast enhancement)
2) Data is split into 80% training and 20% testing
3) SMOTE is applied to balance the data(to overcome unbalanced data)
4) Simple CNN model, where the used parameters are 
    Optimizer - 'adam', loss - 'binary_crossentropy', batch size-32, epochs-10, 

# Evaluations

CNN model achieved 100 % Accuracy on the test data

# User Interface
1) Used streamlit for UI
2) To run the interface (install **streamlit~=1.47.0**), In the terminal write **streamlit run Interface1.py**
3) A new tab will be opened in the browser, where the user can check the urls of images.


