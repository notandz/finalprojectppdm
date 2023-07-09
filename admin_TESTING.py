import os
import glob
import numpy as np
import pandas as pd
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
    
    return np.concatenate(features)

# Train the model
def train_model(k):
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
    model = KNeighborsClassifier(n_neighbors=k)
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

# Streamlit UI - Home Page
def home():
    st.title('Happy or Sad Image Classifier')
    
    # Check if model and dataset files exist
    model_exists = os.path.exists(model_file)
    dataset_exists = os.path.exists(dataset_file)
    
    # Add "Train Model" button
    if st.button("Train Model"):
        st.session_state.page = 'Train'

    # Image classification
    image_folder = st.text_input("Enter the path to the image folder:")
    now = st.selectbox("What data u want to train?", ["Happy", "Sad"])
    if st.button("Classify Images"):
        if not os.path.exists(image_folder):
            st.write("Invalid image folder path!")
            return
        
        image_files = glob.glob(os.path.join(image_folder, '*'))
        if len(image_files) == 0:
            st.write("No images found in the specified folder!")
            return
        
        # Load the model
        if model_exists:
            model = joblib.load(model_file)
            
            # Classify images and create a dataframe
            results = []
            for idx, file in enumerate(image_files):
                img = cv2.resize(cv2.imread(file, 0), (48, 48))
                features = calculate_glcm_features(img)
                prediction = model.predict([features])[0]
                result = {
                    "No.": idx + 1,
                    "Image File": os.path.basename(file),
                    "Classification": "Happy" if prediction == 0 else "Sad"
                }
                
                # Calculate metrics
                if dataset_exists:
                    dataset = np.load(dataset_file)
                    X_test, y_test = dataset[:, :-1], dataset[:, -1]
                    y_pred = model.predict(X_test)
                    precision = metrics.precision_score(y_test, y_pred)
                    recall = metrics.recall_score(y_test, y_pred)
                    f1_score = metrics.f1_score(y_test, y_pred)
                    result["Precision"] = precision
                    result["Recall"] = recall
                    result["F1-Score"] = f1_score
                
                results.append(result)
            
            df = pd.DataFrame(results)
            st.write(df)
            
            
            # Count the number of happy and sad images
            count = df['Classification'].value_counts()
            st.write("Number of Detected as Happy Images:", count["Happy"])
            st.write("Number of Detected as Sad Images:", count["Sad"])
            st.write("Total Images:", len(image_files))
            
            if now == "Happy":
                st.write("Accuracy:", round(count["Happy"] / len(image_files) * 100, 2), "%")
            else:
                st.write("Accuracy:", round(count["Sad"] / len(image_files) * 100, 2), "%")
    
            st.write("Precision:", round(df["Precision"].mean(), 2))
            st.write("Recall:", round(df["Recall"].mean(), 2))
            st.write("F1-Score:", round(df["F1-Score"].mean(), 2))

# Streamlit UI - Train Page
def train():
    st.title('Train Model')
    
    # Add "Back" button
    if st.button("Back"):
        st.session_state.page = 'Home'
    
    # Select K value
    k = st.selectbox("Select K value", [1, 3, 5, 7, 9])
    
    # Train the model
    if st.button("Train", key="train_button"):
        train_model(k)

# Streamlit App
def main():
    # Initialize Streamlit session state
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'
    
    # Page navigation
    if st.session_state.page == 'Home':
        home()
    elif st.session_state.page == 'Train':
        train()

if __name__ == "__main__":
    main()
