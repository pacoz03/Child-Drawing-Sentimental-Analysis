import csv
import matplotlib.pyplot as plt
import os

def log_metrics(log_file, model_name, epochs, train_acc, test_acc, train_loss, test_loss, data_augmentation_params):
    # Salvataggio delle metriche in un file CSV
    header = ['Model', 'Epochs', 'Train Accuracy', 'Test Accuracy', 'Train Loss', 'Test Loss', 'Data Augmentation']
    data = [
        model_name,
        epochs,
        train_acc,
        test_acc,
        train_loss,
        test_loss,
        str(data_augmentation_params)  # Salva i parametri come stringa
    ]

    # Se il file non esiste, crea l'intestazione
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Scrivi le metriche nel file
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def plot_metrics(log_file):
    # Lettura delle metriche e visualizzazione dei risultati
    metrics = []
    with open(log_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics.append(row)

    # Convertire in liste per il plotting
    models = [m['Model'] for m in metrics]
    train_acc = [float(m['Train Accuracy']) for m in metrics]
    test_acc = [float(m['Test Accuracy']) for m in metrics]
    epochs = [int(m['Epochs']) for m in metrics]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(models, train_acc, label='Train Accuracy', marker='o')
    plt.plot(models, test_acc, label='Test Accuracy', marker='o')
    plt.title('Performance del Modello')
    plt.xlabel('Esperimenti')
    plt.ylabel('Accuratezza')
    plt.legend()
    plt.grid()
    plt.show()


