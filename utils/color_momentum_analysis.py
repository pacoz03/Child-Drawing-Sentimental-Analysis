import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Carica il file CSV generato
def load_data(file_path):
    """
    Carica i dati da un file CSV.
    """
    try:
        data = pd.read_csv(file_path)
        print("Dati caricati con successo!")
        print(data.head())
        return data
    except FileNotFoundError:
        print(f"Errore: Il file {file_path} non è stato trovato.")
        return None


def plot_correlation_heatmap(data, save_path=None):
    """
    Calcola e visualizza la matrice di correlazione e salva il grafico se specificato.
    """
    # Seleziona solo le colonne numeriche
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_data.corr()

    # Visualizza la matrice di correlazione
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matrice di correlazione tra le variabili numeriche")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_color_distribution(data, channel='Red_mean', save_path=None):
    """
    Visualizza la distribuzione della media di un canale colore per sentiment e salva il grafico se specificato.
    """
    if 'sentiment' not in data.columns or channel not in data.columns:
        print(f"Errore: 'sentiment' o '{channel}' non trovati nel dataset.")
        return

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='sentiment', y=channel, data=data)
    plt.title(f"Distribuzione di {channel} per sentimento")
    plt.xticks(rotation=45)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def analyze_dataset(file_path):
    """
    Esegue l'analisi completa del dataset salvando i grafici.
    """
    data = load_data(file_path)
    if data is not None:
        # Percorso per salvare i grafici
        output_dir = "C:\\Users\\pasqu\\PycharmProjects\\Child-Drawing-Sentimental-Analysis\\output_graphs\\"
        os.makedirs(output_dir, exist_ok=True)

        # Visualizza e salva la matrice di correlazione
        plot_correlation_heatmap(data, save_path=os.path.join(output_dir, "correlation_heatmap.png"))

        # Visualizza e salva la distribuzione di un canale colore (ad esempio, Red_mean)
        plot_color_distribution(data, channel='Red_mean', save_path=os.path.join(output_dir, "red_mean_distribution1.png"))
        plot_color_distribution(data, channel='Green_mean', save_path=os.path.join(output_dir, "green_mean_distribution1.png"))
        plot_color_distribution(data, channel='Blue_mean', save_path=os.path.join(output_dir, "blue_mean_distribution1.png"))
        plot_color_distribution(data, channel='Hue_mean', save_path=os.path.join(output_dir, "hue_mean_distribution1.png"))
        plot_color_distribution(data, channel='Saturation_mean', save_path=os.path.join(output_dir, "saturation_mean_distribution1.png"))
        plot_color_distribution(data, channel='Value_mean', save_path=os.path.join(output_dir, "value_mean_distribution1.png"))


# Percorso del dataset
file_path = "C:\\Users\\pasqu\\PycharmProjects\\Child-Drawing-Sentimental-Analysis\\color_momentum\\color_moments.csv"

# Analizza il dataset
analyze_dataset(file_path)
