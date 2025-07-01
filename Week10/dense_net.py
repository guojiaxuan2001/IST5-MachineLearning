from tensorflow.keras import datasets, layers, models, applications, optimizers

base_model = applications.DenseNet121(input_shape=(32, 32, 3),
                                      include_top=False, pooling='avg',
                                      weights='imagenet')

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

base_model.trainable = False

for layer in base_model.layers[-60:]:
    layer.trainable = True

new_model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

adam_low_rate = optimizers.Adam(learning_rate=0.001)
new_model.compile(optimizer=adam_low_rate,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

history = new_model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))