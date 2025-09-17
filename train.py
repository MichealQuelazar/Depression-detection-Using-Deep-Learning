import os
import shutil
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Step 1: Download dataset
path = kagglehub.dataset_download("khairunneesa/depression-dataset-on-facial-ecpression-images")
print("Dataset downloaded at:", path)

# Step 2: Locate dataset
original_data_path = os.path.join(path, "Depression Data", "data")
subdirs = ["train", "val", "test"]

# Step 3: Reorganize into binary folders (Depressed vs Non-depressed)
depressed_classes = ["sad", "neutral"]
binary_data_path = "binary_data"
os.makedirs(os.path.join(binary_data_path, "depressed"), exist_ok=True)
os.makedirs(os.path.join(binary_data_path, "non_depressed"), exist_ok=True)

for subdir in subdirs:
    subdir_path = os.path.join(original_data_path, subdir)
    for emotion_folder in os.listdir(subdir_path):
        emotion_folder_path = os.path.join(subdir_path, emotion_folder)
        if not os.path.isdir(emotion_folder_path):
            continue
        if emotion_folder.lower() in depressed_classes:
            target_folder = os.path.join(binary_data_path, "depressed")
        else:
            target_folder = os.path.join(binary_data_path, "non_depressed")

        for img_file in os.listdir(emotion_folder_path):
            src_file = os.path.join(emotion_folder_path, img_file)
            dst_file = os.path.join(target_folder, f"{subdir}_{emotion_folder}_{img_file}")
            if os.path.isfile(src_file):
                shutil.copy(src_file, dst_file)

# Step 4: Data generators
img_height, img_width = 128, 128
batch_size = 32
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    binary_data_path,
    target_size=(img_height, img_width),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    binary_data_path,
    target_size=(img_height, img_width),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# Step 5: Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Step 6: Train
epochs = 10
history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Step 7: Save model
model.save("depression_model.h5")
print("Model saved as depression_model.h5")



model = tf.keras.models.load_model("depression_model.h5", compile=False)


converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: optimize for size and speed
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open("depression_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to depression_model.tflite")