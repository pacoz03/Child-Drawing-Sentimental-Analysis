# Importiamo le librerie necessarie
import cv2  # Per la gestione delle immagini
import numpy as np  # Per operazioni matematiche
import pandas as pd  # Per salvare i dati in un formato tabellare (CSV)
import os  # Per gestire i file e le directory


# Funzione per calcolare i momenti cromatici
def calculate_color_moments(image, color_space='RGB'):
    """
    Calcola i momenti cromatici (media, varianza, skewness) per ciascun canale colore.

    Args:
    - image: Immagine in formato array.
    - color_space: Spazio colore ('RGB' o 'HSV').

    Returns:
    - Un dizionario con i momenti calcolati.
    """
    if color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    moments = {}
    channel_names = ['Red', 'Green', 'Blue'] if color_space == 'RGB' else ['Hue', 'Saturation', 'Value']
    for i, channel in enumerate(cv2.split(image)):  # Divide i canali (R,G,B o H,S,V)
        moments[f'{channel_names[i]}_mean'] = np.mean(channel)
        moments[f'{channel_names[i]}_var'] = np.var(channel)
        moments[f'{channel_names[i]}_skew'] = np.mean((channel - np.mean(channel)) ** 3) / (np.var(channel) ** 1.5 + 1e-6)

    return moments


def process_dataset(dataset_path, output_file):
    """
    Legge le immagini da un dataset organizzato per sentimenti,
    calcola i momenti cromatici e salva i risultati in un CSV.

    Args:
    - dataset_path: Percorso alla directory principale del dataset.
    - output_file: Nome del file CSV dove salvare i dati.
    """
    # Mappatura dei sentimenti a valori numerici
    sentiment_mapping = {
        "Happy": 0,
        "Sad": 1,
        "Angry": 2,
        "Fear": 3,
    }

    data = []  # Lista per memorizzare i risultati

    # Iteriamo su ogni directory (una per ogni sentimento)
    for sentiment in os.listdir(dataset_path):
        sentiment_path = os.path.join(dataset_path, sentiment)

        if not os.path.isdir(sentiment_path):
            continue  # Salta se non è una directory

        # Itera su ogni immagine nella directory del sentimento
        for img_file in os.listdir(sentiment_path):
            img_path = os.path.join(sentiment_path, img_file)

            # Controlla che sia un file immagine
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue

            # Leggi l'immagine
            image = cv2.imread(img_path)
            if image is None:
                print(f"Errore nella lettura dell'immagine: {img_path}")
                continue

            # Calcola i momenti cromatici per RGB e HSV
            rgb_moments = calculate_color_moments(image, color_space='RGB')
            hsv_moments = calculate_color_moments(image, color_space='HSV')

            # Combina i risultati con l'etichetta (sentimento)
            record = {
                "sentiment": sentiment, 
                "sentiment_numeric": sentiment_mapping.get(sentiment, -1),
                "image_name": img_file
            }
            record.update(rgb_moments)
            record.update(hsv_moments)

            # Aggiungi il risultato alla lista
            data.append(record)

    # Converti i dati in un DataFrame e salva in un file CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Dati salvati in: {output_file}")


# Percorso al dataset
dataset_path = "C:\\Users\\pasqu\\PycharmProjects\\Child-Drawing-Sentimental-Analysis\\dataset"  # Cambia con il percorso reale
output_file = "../color_momentum/color_moments.csv"  # File dove salvare i risultati

# Elaborazione del dataset
process_dataset(dataset_path, output_file)