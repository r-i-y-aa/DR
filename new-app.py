import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
#added stuff
train_dir = '/Users/riyak/desktop/DR_DATA2/Training'

train_datagen = ImageDataGenerator(
   rescale=1./255,
   rotation_range=20,
   width_shift_range=0.2,
   height_shift_range=0.2,
   shear_range=0.2,
   zoom_range=0.2,
   horizontal_flip=True,
   fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
   train_dir,
   target_size=(150, 150),
   batch_size=32,
   class_mode='binary'
)

X, y = [], []
for i in range(10):
   img_batch, label_batch = next(train_generator)
   for img, label in zip(img_batch, label_batch):
       X.append(img)
       y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1, 150, 150, 3)
X_val = X_val.reshape(-1, 150, 150, 3)

tree_model = DecisionTreeClassifier(max_depth=200)
tree_model.fit(X_train.reshape(len(X_train), -1), y_train)

y_test_tree = tree_model.predict(X_val.reshape(len(X_val), -1))
train_accuracy_tree = accuracy_score(y_train, tree_model.predict(X_train.reshape(len(X_train), -1)))
accuracy_tree = accuracy_score(y_val, y_test_tree)

print(f"Decision Training Tree Accuracy:{train_accuracy_tree:.4f}")
print(f"Decision Tree Accuracy: {accuracy_tree:.4f}")

forest_model = RandomForestClassifier(n_estimators=100, max_depth=100)
forest_model.fit(X_train.reshape(len(X_train), -1), y_train)

y_test_forest = forest_model.predict(X_val.reshape(len(X_val), -1))
accuracy_forest = accuracy_score(y_val, y_test_forest)

print(f"Random Forest Accuracy: {accuracy_forest:.4f}")

model = Sequential([
   Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
   MaxPooling2D((2, 2)),
   Conv2D(64, (3, 3), activation='relu'),
   MaxPooling2D((2, 2)),
   Conv2D(128, (3, 3), activation='relu'),
   MaxPooling2D((2, 2)),
   Conv2D(128, (3, 3), activation='relu'),
   MaxPooling2D((2, 2)),
   Flatten(),
   Dense(512, activation='relu'),
   Dropout(0.2),
   Dense(512, activation='relu'),
   Dropout(0.2),
   Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
   X_train, y_train,
   validation_data=(X_val, y_val),
   epochs=10,
   batch_size=32
)

model.save('/Users/riyak/desktop/DR_DATA2/diabetic_retinopathy_model.h5')

CNN_model = model.predict(X_val)
rounded_predictions = (CNN_model > 0.5).astype(int)
print("CNN Model f1 score:", f1_score(y_val, rounded_predictions))
print("Decision Tree f1 score:", f1_score(y_val, y_test_tree))
print("Random Forest f1 score:", f1_score(y_val, y_test_forest))

y_prob_tree = tree_model.predict_proba(X_val.reshape(len(X_val), -1))[:, 1]
y_prob_forest = forest_model.predict_proba(X_val.reshape(len(X_val), -1))[:, 1]
y_prob_CNN = CNN_model.flatten()

fpr_tree, tpr_tree, _ = roc_curve(y_val, y_prob_tree)
fpr_forest, tpr_forest, _ = roc_curve(y_val, y_prob_forest)
fpr_CNN, tpr_CNN, _ = roc_curve(y_val, y_prob_CNN)

roc_auc_tree = auc(fpr_tree, tpr_tree)
roc_auc_forest = auc(fpr_forest, tpr_forest)
roc_auc_CNN = auc(fpr_CNN, tpr_CNN)

plt.figure(figsize=(10, 8))
plt.plot(fpr_tree, tpr_tree, color='darkorange', lw=2, label=f'Decision Tree (AUC = {roc_auc_tree:.2f})')
plt.plot(fpr_forest, tpr_forest, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_forest:.2f})')
plt.plot(fpr_CNN, tpr_CNN, color='blue', lw=2, label=f'CNN (AUC = {roc_auc_CNN:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

w1, w2, w3 = 1/3, 1/3, 1/3
finalCorrect = 0
weightList = []

for i in range(len(X_val)):
   if y_test_tree[i] != y_val[i]:
       w1 /= 2
   if y_test_forest[i] != y_val[i]:
       w2 /= 2
   if CNN_model[i] != y_val[i]:
       w3 /= 2

   sum_w = w1 + w2 + w3
   w1 /= sum_w
   w2 /= sum_w
   w3 /= sum_w

   pred = w1 * y_test_tree[i] + w2 * y_test_forest[i] + w3 * CNN_model[i]
   pred = 1 if pred > 0.5 else 0

   if pred == y_val[i]:
       finalCorrect += 1
   weightList.append(pred)

cm = confusion_matrix(y_val, weightList)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

print(finalCorrect)
print(finalCorrect / len(y_val))
print(w1,w2,w3)

plt.plot(history.history['accuracy'])
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train'], loc='upper left')
plt.show()

print("Model training complete and saved!")
