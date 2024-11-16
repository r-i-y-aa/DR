import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

base_dir = '/Users/riyak/desktop/data/training'

# Target image size (e.g., 224x224)
image_size = (224, 224)

# Lists to store the image data and labels
X = []
Y = []

# Get the folder names (which should correspond to class labels)
class_folders = sorted(os.listdir(base_dir))

# Loop over each class folder
for label in class_folders:
    folder_path = os.path.join(base_dir, label)

    if os.path.isdir(folder_path):  # Ensure it's a folder
        # Loop over each image file in the folder
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)

            try:
                # Open the image
                img = Image.open(img_path)

                # Resize the image to the target size
                img = img.resize(image_size)

                # Convert the image to a NumPy array and normalize the pixel values
                img_array = np.array(img) / 255.0

                # Add the image data to X
                X.append(img_array)

                # Add the corresponding label to Y
                Y.append(label)

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

# Convert X and Y to NumPy arrays
X = np.array(X)
Y = np.array(Y)

# Encode labels to integers
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Now, X is the feature array and Y_encoded is the target array with labels as integers
print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {Y_encoded.shape}")

# Number of classes
num_classes = 5

# Input shape - This should match the image size (e.g., 224x224x3 for RGB images)
input_shape = (224, 224, 3)

# Define the
model = Sequential()

# First Convolutional Layer with MaxPooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Layer with MaxPooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolutional Layer with MaxPooling
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output to feed into fully connected layers
model.add(Flatten())

# Fully connected layer
model.add(Dense(512, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output layer with softmax activation (for multiclass classification)
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy for integer labels
              metrics=['accuracy'])

# Summary of the model architecture
model.summary()


# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y_encoded, test_size=0.2)

# Train the model
history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=10,  # You can adjust the number of epochs
                    batch_size=32)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(X_val, Y_val)
print(f"Validation Accuracy: {val_acc}")

# Load the trained model
# model = load_model('retinopathy_model.h5')


# Function to load and preprocess an image
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(img_path)

    # Resize the image to match the input shape expected by the model
    img = img.resize(target_size)

    # Convert the image to a NumPy array
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]

    # If the image has only one channel (grayscale), convert it to 3 channels (RGB)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    # Add a batch dimension (as the model expects batches of images, even for a single image)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

 
# Function to make a prediction
def predict_image_class(img_path, model, class_labels):
    # Preprocess the image
    img_array = load_and_preprocess_image(img_path)

    # Make a prediction
    predictions = model.predict(img_array)

    # Get the index of the highest predicted class
    predicted_class_index = np.argmax(predictions)

    # Get the class label
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label


# Example usage:
# Define the class labels (make sure these correspond to your actual class labels)
class_labels = ['0', '1', '2', '3', '4']

# Path to the image you want to predict
image_path = '/Users/riyak/desktop/data/test/2/2081_left.jpeg'

# Get the predicted class
predicted_class = predict_image_class(image_path, model, class_labels)

print(f"The predicted class is: {predicted_class}")

#Normalize images (width, height, color channels, rotation, dimensions)
#Split data into train, val, test
#Create a model or set of models
#Compile the Models (define number of epochs, optimizer, learning rate)
#Predict on the validation set
#Plot the accuracy
# **Multiplicative weight update to combine a set of models**
#Connect this program with the other one

#Accuracy of the two models is the product of the two models' accuracies