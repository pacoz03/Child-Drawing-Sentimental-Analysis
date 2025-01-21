import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

dataset_path = 'C:\\Users\\pasqu\\PycharmProjects\\Child-Drawing-Sentimental-Analysis\\dataset\\'
emotions = ['Angry', 'Fear', 'Happy', 'Sad']




def mostra_immagine(percorso_immagine):
    if not os.path.exists(percorso_immagine):
        print(f"Errore: il file {percorso_immagine} non esiste.")
        return
    img = cv2.imread(percorso_immagine)
    if img is None:
        print(f"Errore: impossibile leggere l'immagine {percorso_immagine}.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def converti_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def calcola_istogramma(img_hsv):
    hist_hue = cv2.calcHist([img_hsv], [0], None, [180], [0, 180])
    hist_sat = cv2.calcHist([img_hsv], [1], None, [256], [0, 256])
    hist_val = cv2.calcHist([img_hsv], [2], None, [256], [0, 256])
    return hist_hue, hist_sat, hist_val

def mostra_istogrammi(hist_hue, hist_sat, hist_val):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(hist_hue, color='r')
    plt.title('Istogramma Tonalità')
    plt.subplot(1, 3, 2)
    plt.plot(hist_sat, color='g')
    plt.title('Istogramma Saturazione')
    plt.subplot(1, 3, 3)
    plt.plot(hist_val, color='b')
    plt.title('Istogramma Valore')
    plt.show()

def media_istogrammi(cartella):
    hist_hue_tot = np.zeros((180, 1))
    hist_sat_tot = np.zeros((256, 1))
    hist_val_tot = np.zeros((256, 1))
    num_images = len(os.listdir(cartella))

    for file_name in os.listdir(cartella):
        img_path = os.path.join(cartella, file_name)
        img = cv2.imread(img_path)
        img_hsv = converti_hsv(img)
        hist_hue, hist_sat, hist_val = calcola_istogramma(img_hsv)
        hist_hue_tot += hist_hue
        hist_sat_tot += hist_sat
        hist_val_tot += hist_val

    return hist_hue_tot / num_images, hist_sat_tot / num_images, hist_val_tot / num_images


if __name__ == '__main__':
    for emotion in emotions:
        folder_path = os.path.join(dataset_path, emotion)
        num_images = len(os.listdir(folder_path))
        print(f'Numero di immagini in {emotion}: {num_images}')

    # Esempio di visualizzazione di un'immagine
    esempio_percorso = os.path.join(dataset_path, 'Sad', 's106.jpeg')
    mostra_immagine(esempio_percorso)

    # Esempio di utilizzo
    img = cv2.imread(esempio_percorso)
    img_hsv = converti_hsv(img)
    hist_hue, hist_sat, hist_val = calcola_istogramma(img_hsv)
    mostra_istogrammi(hist_hue, hist_sat, hist_val)

    # Calcolo e visualizzazione per ciascuna emozione
    for emotion in emotions:
        folder_path = os.path.join(dataset_path, emotion)
        hist_hue_avg, hist_sat_avg, hist_val_avg = media_istogrammi(folder_path)
        print(f'Istogrammi medi per {emotion}:')
        mostra_istogrammi(hist_hue_avg, hist_sat_avg, hist_val_avg)

