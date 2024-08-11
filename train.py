import splitfolders
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Flatten
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

# splitting into train, validation and test
input_folder = "landmark_data"
splitfolders.ratio(input_folder, output="dataset_II", seed=42, ratio=(.7, .0, .3))

# Define the path to your dataset
train_dataset_directory = "dataset_II/train"
validation_dataset_directory = "dataset_II/test"

# normalization
# normTrain_layer = ImageDataGenerator(rescale=1 / 255)
normTrain_layer = ImageDataGenerator(
    rescale=1/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

normVal_layer = ImageDataGenerator(rescale=1 / 255)
# creating training dataset and val dataset
train_ds = normTrain_layer.flow_from_directory(train_dataset_directory,
                                               target_size=(300, 300),
                                               batch_size=32,
                                               shuffle=True,
                                               class_mode='categorical')
val_ds = normVal_layer.flow_from_directory(validation_dataset_directory,
                                           target_size=(300, 300),
                                           batch_size=32,
                                           class_mode='categorical')

# Define the model correctly
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(26, activation='softmax'))

# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(26, activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# fitting training data and validation data into the model
# for avoiding overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_fit = model.fit(train_ds,
                      epochs=10,
                      callbacks=[early_stopping],
                      validation_data=val_ds)
# Assuming `model` is your trained model
model.save('model/ISL_model1_Full6.h5')
