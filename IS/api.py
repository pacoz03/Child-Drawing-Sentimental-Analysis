from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from enum import Enum
import logging

app = Flask(__name__)

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carica il modello Keras
try:
    model = load_model('miglior_modello.keras')  # Assicurati che 'model.h5' sia nel percorso corretto
    logger.info("Modello Keras caricato correttamente.")
except Exception as e:
    logger.error(f"Errore nel caricamento del modello Keras: {e}")
    model = None

class ValutazioneEmotiva(Enum):
    ARRABBIATO = "ARRABBIATO"
    SPAVENTATO = "SPAVENTATO"
    FELICE = "FELICE"
    TRISTE = "TRISTE"

# Mappa degli indici delle classi alle valutazioni emotive
CLASS_INDEX_MAP = {
    0: ValutazioneEmotiva.ARRABBIATO.value,
    1: ValutazioneEmotiva.SPAVENTATO.value,
    2: ValutazioneEmotiva.FELICE.value,
    3: ValutazioneEmotiva.TRISTE.value
}

@app.route('/valutadisegno', methods=['POST'])
def valuta_disegni():
    if model is None:
        logger.error("Modello non disponibile.")
        return jsonify({"error": "Modello non disponibile."}), 500

    data = request.get_json()
    if not data:
        logger.error("Nessun dato ricevuto.")
        return jsonify({"error": "Nessun dato ricevuto."}), 400

    risultati = []

    for disegno in data:
        disegno_id = disegno.get('disegnoId')
        immagine_base64 = disegno.get('immagine')

        if disegno_id is None or immagine_base64 is None:
            logger.warning(f"Disegno mancante di 'disegnoId' o 'immagine': {disegno}")
            risultati.append({
                'disegnoId': disegno_id,
                'valutazioneEmotiva': 'ERRORE'
            })
            continue

        try:
            # Decodifica l'immagine
            immagine_bytes = base64.b64decode(immagine_base64)
            immagine = Image.open(io.BytesIO(immagine_bytes)).convert('RGB')
            immagine = immagine.resize((224, 224))  # Adatta la dimensione secondo il tuo modello
            immagine_array = np.array(immagine) / 255.0
            immagine_array = np.expand_dims(immagine_array, axis=0)

            # Previsione con il modello
            predizione = model.predict(immagine_array)
            classe = np.argmax(predizione, axis=1)[0]

            # Mappa la classe alla ValutazioneEmotiva
            valutazione = CLASS_INDEX_MAP.get(classe, 'ERRORE')

            risultati.append({
                'disegnoId': disegno_id,
                'valutazioneEmotiva': valutazione
            })

            logger.info(f"Disegno ID {disegno_id} valutato: {valutazione}")

        except Exception as e:
            logger.error(f"Errore nel processare Disegno ID {disegno_id}: {e}")
            risultati.append({
                'disegnoId': disegno_id,
                'valutazioneEmotiva': 'ERRORE'
            })

    return jsonify(risultati), 200

if __name__ == '__main__':
    app.run(port=6000)
