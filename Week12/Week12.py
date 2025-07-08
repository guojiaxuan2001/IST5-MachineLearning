import tensorflow as tf
from tensorflow.keras import layers, models, datasets, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.squeeze(), y_test.squeeze()

datagen = ImageDataGenerator(
    horizontal_flip = True,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
)

datagen.fit(x_train)


def res_block(inputs, filters, stride=1):
    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, use_bias=False)(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_resnet():
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(16, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)


    for filters, strides in zip([16]*3 + [32]*3 + [64]*3,
                                [1]*3 + [2] + [1]*2 + [2] + [1]*2):
        x = res_block(x, filters, stride=strides)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model
opt1 = optimizers.Adam(1e-3)
opt2 = optimizers.SGD(learning_rate = 1e-2, momentum = 0.9)

def train_with_optimizer(optimizer, label):
    print(f"\n Training with {label}...\n")
    model = build_resnet()
    model.compile(optimizer = optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=10,
                        batch_size=64,
                        verbose = 2)
    return history

history_adam = train_with_optimizer(opt1, 'Adam')
history_SGD = train_with_optimizer(opt2, 'SGD')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_adam.history['val_accuracy'], label='Adam Val Acc', linestyle='-', marker='o')
plt.plot(history_SGD.history['val_accuracy'], label='SGD Val Acc', linestyle='--', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show