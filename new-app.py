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

# Define paths to your directories
train_dir = '/Users/riyak/desktop/Diabetic_Retinopathy/Main/train'

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

# Decision Tree Classifier
tree_model = DecisionTreeClassifier( max_depth=90)
tree_model.fit(X_train, y_train)

# Make predictions on the validation set
y_test_tree = tree_model.predict(X_val)

# Calculate the accuracy for Decision Tree
accuracy_tree = accuracy_score(y_val, y_test_tree)
print(f"Decision Tree Accuracy: {accuracy_tree:.4f}")

# Random Forest Classifier
forest_model = RandomForestClassifier(n_estimators=100, max_depth=1000)
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
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the CNN model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN model
history = model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // train_generator.batch_size
)

# CNN_model = model.predict(X_val)

# Plot training accuracy for CNN
plt.plot(history.history['accuracy'])
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train'], loc='upper left')
plt.show()

w1, w2, w3 = 1/3
for i in range(len(X_val)):
    if (y_test_tree[i]!=y_val[i]):
        w1/=2
    if(y_test_forest[i]!=y_val[i]):
        w2/=2
    # if(CNN_model[i]!=y_val[i]):
    #     w3/=2
    sum = w1+w2
    w1/=sum
    w2/=sum
    # w3/=sum
    y1 = y_test_tree
    y2 = y_test_forest
    # y3 = CNN_model
    pred = w1*y1+w2*y2
    if (pred > 0.5):
        pred = 1
    else:
        pred = 0
    print(pred, y_val[i])
# Save the CNN model
# model.save('/Users/riyak/desktop/Diabetic_Retinopathy/diabetic_retinopathy_model.h5')

print("Model training complete and saved!")