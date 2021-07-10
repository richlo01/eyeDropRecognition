import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobileNetV2_preprocess
from glob import glob



dataPath = './data'

folders = glob(dataPath + '/*')
classCount = len(folders)
print(f"Classes: {classCount}")

DIM = 224



dropout = 0.2
baseModel = MobileNetV2(
    input_shape = (DIM, DIM, 3),
    include_top=False, weights='imagenet',
)
baseModel.trainable = False


model = Sequential([baseModel,
                              GlobalAveragePooling2D(),
                              Dropout(dropout),
                              Dense(classCount, activation="softmax")
                            ])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


augmentation = ImageDataGenerator(
    rotation_range=200,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.5,
    brightness_range=[0.2,1.1],
    channel_shift_range=10,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=mobileNetV2_preprocess,
    validation_split=0.05,
)


#Try using channel shifts


batch_size = 16
trainingGen = augmentation.flow_from_directory(
        dataPath,
        target_size=(DIM, DIM),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True)

validationGen = augmentation.flow_from_directory(
        dataPath,
        target_size=(DIM, DIM),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True)


epochs = 80
f = model.fit_generator(
      trainingGen,
      steps_per_epoch = trainingGen.samples // batch_size,
      validation_data = validationGen,
      validation_steps = validationGen.samples // batch_size,
      epochs = epochs,
      verbose=1)

testAugmentation = ImageDataGenerator(
    preprocessing_function=mobileNetV2_preprocess,
)


testGen = testAugmentation.flow_from_directory(
        dataPath,
        target_size=(DIM, DIM),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

eval = model.evaluate_generator(
  testGen,
  100,
  verbose=1,
)

print(eval)

model.save(f"{batch_size}_bs_{epochs}_epochs_{dropout}_droput_{eval[1]}_acc.h5")