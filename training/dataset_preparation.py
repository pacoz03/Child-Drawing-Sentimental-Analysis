import os
import shutil
import random

# Elenco delle classi (corrispondenti alle sottocartelle)
categories = ['Angry', 'Fear', 'Happy', 'Sad']

def train_test_split(base_dir, split_dir):
    # Definiamo le percentuali di split
    train_ratio = 0.7
    test_ratio = 0.3

    # Per ogni classe (Angry, Fear, Happy, Sad)
    for cat in categories:
        # Percorso sorgente della classe
        cat_path = os.path.join(base_dir, cat)

        # Leggiamo tutti i file della cartella
        images = os.listdir(cat_path)

        # Mischiamo in modo casuale (random shuffle)
        random.shuffle(images)

        # Creiamo le directory di output (train/test) per questa classe
        train_cat_dir = os.path.join(split_dir, 'train', cat)
        test_cat_dir = os.path.join(split_dir, 'test', cat)

        os.makedirs(train_cat_dir, exist_ok=True)
        os.makedirs(test_cat_dir, exist_ok=True)

        # Calcoliamo quanti file vanno in train e test
        num_images = len(images)
        train_count = int(train_ratio * num_images)

        # Dividiamo gli array con slicing
        train_files = images[:train_count]
        test_files = images[train_count:]

        # Copia dei file nella cartella train
        for f in train_files:
            src = os.path.join(cat_path, f)
            dst = os.path.join(train_cat_dir, f)
            shutil.copy(src, dst)

        # Copia dei file nella cartella test
        for f in test_files:
            src = os.path.join(cat_path, f)
            dst = os.path.join(test_cat_dir, f)
            shutil.copy(src, dst)

        # Stampa il numero di file per ogni split
        print(f'Classe {cat}: {len(train_files)} train, {len(test_files)} test')

def test_train_validation_split(base_dir, split_dir):
    # Definiamo le percentuali di split
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # Per ogni classe (Angry, Fear, Happy, Sad)
    for cat in categories:
        # Percorso sorgente della classe
        cat_path = os.path.join(base_dir, cat)

        # Leggiamo tutti i file della cartella
        images = os.listdir(cat_path)

        # Mischiamo in modo casuale (random shuffle)
        random.shuffle(images)

        # Creiamo le directory di output (train/val/test) per questa classe
        train_cat_dir = os.path.join(split_dir, 'train', cat)
        val_cat_dir = os.path.join(split_dir, 'val', cat)
        test_cat_dir = os.path.join(split_dir, 'test', cat)

        os.makedirs(train_cat_dir, exist_ok=True)
        os.makedirs(val_cat_dir, exist_ok=True)
        os.makedirs(test_cat_dir, exist_ok=True)

        # Calcoliamo quanti file vanno in train, val e test
        num_images = len(images)
        train_count = int(train_ratio * num_images)
        val_count = int(val_ratio * num_images)
        test_count = num_images - train_count - val_count

        # Dividiamo gli array con slicing
        train_files = images[:train_count]
        val_files = images[train_count:train_count + val_count]
        test_files = images[train_count + val_count:]

        # Copia dei file nella cartella train
        for f in train_files:
            src = os.path.join(cat_path, f)
            dst = os.path.join(train_cat_dir, f)
            shutil.copy(src, dst)

        # Copia dei file nella cartella val
        for f in val_files:
            src = os.path.join(cat_path, f)
            dst = os.path.join(val_cat_dir, f)
            shutil.copy(src, dst)

        # Copia dei file nella cartella test
        for f in test_files:
            src = os.path.join(cat_path, f)
            dst = os.path.join(test_cat_dir, f)
            shutil.copy(src, dst)

        # Stampa il numero di file per ogni split
        print(f'Classe {cat}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test')
