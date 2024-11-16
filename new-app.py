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
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint

# Define paths to your directories
train_dir = '/Users/riyak/desktop/DR_DATA2/Training'

# ImageDataGenerator for loading and augmenting images
train_datagen = ImageDataGenerator(
    rescale=1./255,               # Normalize images to the range [0, 1]
    rotation_range=20,            # Randomly rotate images
    width_shift_range=0.2,        # Randomly shift images horizontally
    height_shift_range=0.2,       # Randomly shift images vertically
    shear_range=0.2,              # Randomly apply shearing transformations
    zoom_range=0.2,               # Randomly zoom into images
    horizontal_flip=True,         # Randomly flip images horizontally
    fill_mode='nearest'           # Strategy for filling in newly created pixels
)

# Create a generator for the training images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),       # Resize images to 150x150
    batch_size=32,                # Number of images to return in each batch
    class_mode='binary'           # Binary classification mode
)

# Flatten the images for Decision Tree and Random Forest (this works if you have data as numpy arrays)
X, y = [], []
for i in range(len(train_generator)):
    img_batch, label_batch = next(train_generator)  # Use next() correctly here
    for img, label in zip(img_batch, label_batch):
        X.append(img.flatten())  # Flatten each image
        y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset for decision tree and random forest
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Decision Tree Classifier
tree_model = DecisionTreeClassifier( max_depth=100)
tree_model.fit(X_train, y_train)

# # Make predictions on the validation set
y_test_tree = tree_model.predict(X_val)
y_train_tree = tree_model.predict(X_train)
# Calculate the accuracy for Decision Tree
train_accuracy_tree = accuracy_score(y_train, y_train_tree)
accuracy_tree = accuracy_score(y_val, y_test_tree)

print(f"Decision Training Tree Accuracy:{train_accuracy_tree:.4f}")
print(f"Decision Tree Accuracy: {accuracy_tree:.4f}")

# Random Forest Classifier
forest_model = RandomForestClassifier(n_estimators=50, max_depth=100)
forest_model.fit(X_train, y_train)

# Make predictions on the validation set
y_test_forest = forest_model.predict(X_val)

# Calculate the accuracy for Random Forest
accuracy_forest = accuracy_score(y_val, y_test_forest)
print(f"Random Forest Accuracy: {accuracy_forest:.4f}")

# CNN Model Definition (for comparison)
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(512, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(512, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(512, activation=  'relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Define the callback to save the best model based on validation accuracy or loss
# checkpoint = ModelCheckpoint(
#     'best_model.h5',          # Path to save the best model
#     monitor='val_loss',        # Metric to monitor (you can also use 'val_accuracy')
#     save_best_only=True,       # Save only the best model
#     mode='min',                # 'min' for minimizing the loss, 'max' for maximizing accuracy
#     verbose=1                  # Verbosity mode
# )
#Prints number of parameters the model has
model.summary()
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN model with the checkpoint callback
history = model.fit(
    train_generator,
    epochs=20,
    # steps_per_epoch=train_generator.samples // train_generator.batch_size,
    # validation_data=validation_generator,  # Assuming you have a validation generator
    # validation_steps=validation_generator.samples // validation_generator.batch_size,
    # callbacks=[checkpoint]  # Include the checkpoint callback here
)

# Save the CNN model
model.save('/Users/riyak/desktop/DR_DATA2/diabetic_retinopathy_model.h5')
# Reshape X_val for CNN model prediction (assuming X_val is of shape (num_samples, 150, 150, 3))
X_val_reshaped = X_val.reshape(-1, 150, 150, 3)

# Predict with CNN model
CNN_model = model.predict(X_val_reshaped)
CNN_model = CNN_model.flatten()  # Flatten to align with y_test_tree and y_test_forest

# Corrected ensemble prediction loop
w1 = 1/3
w2 = 1/3
w3 = 1/3
finalCorrect = 0
for i in range(len(X_val)):
    if y_test_tree[i] != y_val[i]:
        w1 /= 2
    if y_test_forest[i] != y_val[i]:
        w2 /= 2
    if CNN_model[i] != y_val[i]:
        w3 /= 2

    # Normalize weights
    sum_w = w1 + w2 + w3
    w1 /= sum_w
    w2 /= sum_w
    w3 /= sum_w

    # Ensemble prediction
    y1 = y_test_tree[i]
    y2 = y_test_forest[i]
    y3 = CNN_model[i]
    pred = w1 * y1 + w2 * y2 + w3 * y3

    # Threshold for binary classification
    pred = 1 if pred > 0.5 else 0

    if pred==y_val[i]:
        finalCorrect+=1
    print(pred, y_val[i])
    print(w1, w2, w3)
print(finalCorrect)
print(finalCorrect/len(y_val))
#
# # Plot training accuracy for CNN
# plt.plot(history.history['accuracy'])
# plt.title('CNN Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(['Train'], loc='upper left')
# plt.show()
# #
# Function to load and preprocess an image
# def load_and_preprocess_image(img_path, target_size=(224, 224)):
#     # Load the image
#     img = Image.open(img_path)
# #
# #     # Resize the image to match the input shape expected by the model
#     img = img.resize(target_size)
# #
# #     # Convert the image to a NumPy array
#     img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
# #
# #     # If the image has only one channel (grayscale), convert it to 3 channels (RGB)
#     if len(img_array.shape) == 2:
#         img_array = np.stack((img_array,) * 3, axis=-1)
# #
# #     # Add a batch dimension (as the model expects batches of images, even for a single image)
#     img_array = np.expand_dims(img_array, axis=0)
#
#     return img_array
#
# #
# # # Function to make a prediction
# def predict_image_class(img_path, model, class_labels):
#     # Preprocess the image
#     img_array = load_and_preprocess_image(img_path)
#
# #     # Make a prediction
#     predictions = model.predict(img_array)
# #
# #     # Get the index of the highest predicted class
#     predicted_class_index = np.argmax(predictions)
# #
# #     # Get the class label
#     predicted_class_label = class_labels[predicted_class_index]
# #
#     return predicted_class_label
# #
# #
# # # Example usage:
# # # Define the  class labels (make sure these correspond to your actual class labels)
# class_labels = ['0', '1']
# #
# # # Path to the image you want to predict
# image_path = '/Users/riyak/desktop/DR_DATA1/Training/DR/15_left.jpeg'
# #
# # # Get the predicted class
# predicted_class = predict_image_class(image_path, model, class_labels)
# #
# print(f"The predicted class is: {predicted_class}")
print("Model training complete and saved!")