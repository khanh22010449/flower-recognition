import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

try:
    with h5py.File('data_flowers.h5', 'r') as f:
        print(f.keys())
        data = f['dataset'][:]
        label = f['labels'][:]
        labels_bytes = label.astype('S')
        labels = labels_bytes.astype(int)
except OSError as e:
    print("Error opening file:", e)
    
classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

model = MobileNet(input_shape=(224,224,3), include_top=True)

model.summary()

vector = model.get_layer("reshape_2").output

feature_extractor = tf.keras.Model(model.input, vector)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img_arr = image.img_to_array(img)
    img_arr_b = np.expand_dims(img_arr, axis=0)
    input_img = preprocess_input(img_arr_b)
    feature_vec = feature_extractor.predict(input_img)
    predictions = knn.predict(feature_vec)[0]

    class_name = classes[predictions]

    cv2.putText(frame, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Flower Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
