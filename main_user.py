import os
import glob
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import joblib
import streamlit as st

# Global variables
happy_dir = 'happy'
sad_dir = 'sad'
dataset_file = 'dataset.npy'
model_file = 'model.joblib'

# Calculate GLCM features
def calculate_glcm_features(img):
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    glcm = graycomatrix(img, [1], angles, 256, symmetric=True, normed=True)
    
    features = []
    
    for prop in properties:
        feature = graycoprops(glcm, prop).ravel()
        features.append(feature)
        
        st.write(f"{prop}: {feature}")
    
    return np.concatenate(features)

# Train the model
def train_model():
    st.write("Training Model...")
    
    # Load image files and labels
    happy_files = glob.glob(os.path.join(happy_dir, '*'))
    sad_files = glob.glob(os.path.join(sad_dir, '*'))
    files = happy_files + sad_files
    labels = [0] * len(happy_files) + [1] * len(sad_files)  # 0 for happy, 1 for sad
    
    # Preprocess images and extract GLCM features
    features = [calculate_glcm_features(cv2.resize(cv2.imread(file, 0), (48, 48))) for file in files]

    # Ensure features and labels are of the same size
    features, labels = np.array(features), np.array(labels)
    assert features.shape[0] == labels.shape[0], "Mismatch in features and labels sizes"

    # Train KNN classifier
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    
    st.write(f"Model accuracy: {accuracy*100:.2f}%")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-Score: {f1_score:.2f}")
    
    # Save the model
    joblib.dump(model, model_file)
    st.write("Model saved!")
    
    # Save the dataset
    dataset = np.column_stack((features, labels))
    np.save(dataset_file, dataset)
    st.write("Dataset saved!")

# Correct misclassification
def correct_misclassification(features, correct_label):
    # Load the dataset file
    dataset = np.load(dataset_file)
    labels = dataset[:, -1]  # Extract the labels

    # Find the index of the misclassified image in the dataset
    misclassified_index = np.where(np.all(dataset[:, :-1] == features, axis=1))[0][0]

    # Update the label
    labels[misclassified_index] = correct_label

    # Save the updated dataset
    np.save(dataset_file, np.column_stack((dataset[:, :-1], labels)))

# Streamlit UI
# Streamlit UI
def main():
    st.set_page_config(layout="wide")
    st.markdown(
        """
        <style>
        .container {
            display: flex;
        }
        .left_column {
            width: 50%;
            padding: 1rem;
        }
        .right_column {
            width: 50%;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title('Happy or Sad Image Classifier')

    # Check if model and dataset files exist
    model_exists = os.path.exists(model_file)
    dataset_exists = os.path.exists(dataset_file)

    # Add "Retrain" button
    if st.button("Retrain"):
        train_model()
        model_exists = True

    # Image classification
    file = st.file_uploader("Upload an image")
    col1, col2 = st.columns(2)
    if file is not None:
        # Preprocess and extract GLCM features
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        features = calculate_glcm_features(img)
        
        col1.image(img, caption="Uploaded Image", use_column_width=False)

        if model_exists:
            # Load the model and predict
            model = joblib.load(model_file)
            prediction = model.predict([features])[0]
            if prediction == 0:
                col2.write("The image is happy!")
            else:
                col2.write("The image is sad!")
                
            # Calculate metrics
            if dataset_exists:
                dataset = np.load(dataset_file)
                X_test, y_test = dataset[:, :-1], dataset[:, -1]
                y_pred = model.predict(X_test)
                accuracy = metrics.accuracy_score(y_test, y_pred)
                precision = metrics.precision_score(y_test, y_pred)
                recall = metrics.recall_score(y_test, y_pred)
                f1_score = metrics.f1_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy*100:.2f}%")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")
                st.write(f"F1-Score: {f1_score:.2f}")

            # Classification result and correction
            if st.button("Correct"):
                if prediction == 0:
                    correct_misclassification(features, 1)
                else:
                    correct_misclassification(features, 0)
                st.write("Misclassification corrected!")

if __name__ == "__main__":
    main()
