import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
from imgaug import augmenters as iaa

# Define augmentations
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),  # Flip horizontally
    iaa.Affine(rotate=(-30, 30)),  # Rotate between -30 to 30 degrees
    iaa.Multiply((0.8, 1.2)),  # Brightness adjustment
])

# Function to extract color histogram features
def extract_color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Function to load images, apply augmentation, and extract features
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
                    
                    # Apply augmentation
                    augmented_image = augmentation(images=[image])[0]
                    
                    # Extract features
                    features.append(extract_color_histogram(augmented_image))
                    labels.append(label)
    return np.array(features), np.array(labels)

# Load dataset
data_dir = "Paddy Phase"  # Replace with your dataset folder
features, labels = load_images_and_labels(data_dir)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define hyperparameter grid for RandomForest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search to optimize hyperparameters
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
model_file = "paddy_phase_classifier_rf.joblib"
joblib.dump(best_model, model_file)
print(f"Model saved to {model_file}")
