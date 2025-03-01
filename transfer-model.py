from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from PIL import Image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

base_dir = '/Users/riyak/desktop/data/training'
# Target image size (e.g., 150x150)
image_size = (150, 150)

# Lists to store the image data and labels
X = []
Y = []
#added stuff
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

# Input shape - This should match the image size (e.g., 150x150x3 for RGB images)
input_shape = (150, 150, 3)
inputs = Input(shape=input_shape)
num_classes = len(np.unique(Y_encoded))
print(f"Number of classes: {num_classes}")

# Apply augmentation to the training data manually
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

# Convert labels to one-hot encoding
Y_one_hot = to_categorical(Y_encoded, num_classes=num_classes)

# Split the dataset into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y_one_hot, test_size=0.2, random_state=42)

# Normalize the pixel values to [0, 1]
X_train /= 255.0
X_val /= 255.0

# Define the model architecture
x = Conv2D(64, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(512, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)  # Multi-class classification

# Create the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the custom dataset
history = model.fit(
    X_train, Y_train,            # Training data
    epochs=20,                   # Adjust based on your needs
    batch_size=32,               # Number of samples per batch
    validation_data=(X_val, Y_val)  # Validation data
)

# Save the trained model
model.save('trained_model.h5')

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, Y_val, verbose=1)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
