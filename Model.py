import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam


train_dir = "F:\Rice_leaf_Disease\split_leaf\\train"
test_dir = "F:\Rice_leaf_Disease\split_leaf\\test"
val_dir = "F:\Rice_leaf_Disease\split_leaf\\validation"

img_size = (224,224)
batch_size = 32
num_classes = 4
learning_rate = 0.001


train_datagen = ImageDataGenerator(rescale = 1.0/255.0)
val_datagen = ImageDataGenerator(rescale = 1.0/255.0)
test_datagen = ImageDataGenerator(rescale = 1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle= False
)

input_shape = (224,224,3)
num_class = 4

model = Sequential()
densenet_base = DenseNet121(weights="imagenet",include_top = False,input_shape = input_shape)
model.add(densenet_base)
model.add(GlobalAveragePooling2D())   # this can be used instead of flatten layer


model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(num_class,activation = 'softmax'))

model.compile(optimizer = Adam(learning_rate=0.001),
loss = tf.keras.losses.CategoricalCrossentropy(),
metrics = ['accuracy'])

# print(model.summary())

history = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs = 50,
    steps_per_epoch = train_generator.samples//batch_size,
    validation_steps = val_generator.samples//batch_size
)


test_loss,test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy* 100:.2f}")
model.save("Disease_dtct.h5")
print("model saved successfully")