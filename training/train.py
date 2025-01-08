import tensorflow as tf
from keras.src.applications.vgg16 import VGG16
from keras.src.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import dataset_preparation
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import os
import metrics

def data_augmentation(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True):
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=rescale,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
    )
    return train_datagen

def base_train():
    epochs = 5
    # 1. Definisci il path per i dati (cartelle train/val/test)
    split_dir = 'C:\\Users\\pasqu\\PycharmProjects\\Child-Drawing-Sentimental-Analysis\\dataset\\NewArts2_splitted1'
    dataset_preparation.train_test_split(
        split_dir=split_dir,
        base_dir='C:\\Users\\pasqu\\PycharmProjects\\Child-Drawing-Sentimental-Analysis\\dataset\\')

    train_dir = os.path.join(split_dir, 'train')
    test_dir  = os.path.join(split_dir, 'test')

    # 2. Data augmentation e generator
    #1 primo train tutto a 0.1 e ho ottenuto 8,19 40 epoch
    #2 0.2 0.2 .1 .3 Test Loss: 0.8227, Test Accuracy: 0.6646 50 epoch
    #3 tutto a 0.1 Test Loss: 0.4843, Test Accuracy: 0.9130 70 epoch
    train_datagen = data_augmentation(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    test_datagen= ImageDataGenerator(rescale=1./255)

    batch_size = 16
    img_size = (224, 224)  # Dimensione di input

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # di solito test non si mescola
    )

    # 3. Definizione di una CNN semplice (da zero)
    # Creazione del modello
    model = Sequential()

    # Primo blocco convoluzionale
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Secondo blocco convoluzionale
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Terzo blocco convoluzionale
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Appiattimento
    model.add(Flatten())

    # Primo livello denso
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))  # Dropout per prevenire overfitting

    # Secondo livello denso
    model.add(Dense(128, activation='relu'))

    # Livello di output
    model.add(Dense(4, activation='softmax'))  # 4 classi nel dataset

    # Compilazione del modello
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    #model.summary()

    # 5. Training con EarlyStopping -- si ferma troppo presto nel training
    early_stop = EarlyStopping(monitor='accuracy', patience=3, restore_best_weights=True)

    # 5. Training con ModelCheckpoint
    checkpoint = ModelCheckpoint(
        filepath='miglior_modello.keras',  # Percorso per salvare il modello
        monitor='accuracy',             # Metrica da monitorare
        save_best_only=True,            # Salva solo il modello con le migliori prestazioni
        mode='max',                     # Modalità 'min' per minimizzare la perdita
        verbose=1                       # Visualizza messaggi durante il salvataggio
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        callbacks=[checkpoint]
    )

    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


def train_with_vgg():
    epochs = 5
    # 1. Definisci il path per i dati (cartelle train/val/test)
    split_dir = 'C:\\Users\\pasqu\\PycharmProjects\\Child-Drawing-Sentimental-Analysis\\dataset\\NewArts2_splitted1'
    dataset_preparation.train_test_split(
        split_dir=split_dir,
        base_dir='C:\\Users\\pasqu\\PycharmProjects\\Child-Drawing-Sentimental-Analysis\\dataset\\')

    train_dir = os.path.join(split_dir, 'train')
    test_dir  = os.path.join(split_dir, 'test')

    # 2. Data augmentation e generator
    train_datagen = data_augmentation(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    batch_size = 16
    img_size = (224, 224)  # Dimensione di input per VGG16

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # 3. Carica il modello VGG16 pre-addestrato
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Congela i pesi del modello pre-addestrato
    base_model.trainable = False

    # Costruisci il modello
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(4, activation='softmax')  # 4 classi nel dataset
    ])

    # Compilazione del modello
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    #model.summary()

    # 5. Training con ModelCheckpoint
    checkpoint = ModelCheckpoint(
        filepath='miglior_modello_vgg.keras',  # Percorso per salvare il modello
        monitor='accuracy',                   # Metrica da monitorare
        save_best_only=True,                  # Salva solo il modello con le migliori prestazioni
        mode='max',                           # Modalità 'max' per massimizzare l'accuratezza
        verbose=1                             # Visualizza messaggi durante il salvataggio
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        callbacks=[checkpoint]
    )

    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

#Esegui
if __name__ == '__main__':
    train_with_vgg()