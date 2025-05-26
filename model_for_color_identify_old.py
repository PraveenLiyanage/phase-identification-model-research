import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the model

# Function to load images and labels
def load_images_and_labels(data_dir):
    labels = []
    features = []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                # Read and resize the image
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.resize(image, (128, 128))
                    features.append(extract_color_features(image))
                    labels.append(label)
    return np.array(features), np.array(labels)

# Function to extract color features
def extract_color_features(image):
    mean_colors = np.mean(image, axis=(0, 1))  # Mean of R, G, B
    std_colors = np.std(image, axis=(0, 1))   # Std dev of R, G, B
    return np.concatenate([mean_colors, std_colors])

# Load dataset
data_dir = "Paddy Phase"
features, labels = load_images_and_labels(data_dir)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model to a file
model_file = "paddy_phase_classifier.joblib"
joblib.dump(model, model_file)
print(f"Model saved to {model_file}")
