import kagglehub
def download_dataset():
    dataset_path = kagglehub.dataset_download("babyhackershal2020/children-drawings")
    print(f"Dataset scaricato in: {dataset_path}")

download_dataset()

