import os
import tensorflow as tf
import pickle
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

# === Définition des couches custom ===
class EuclideanDistanceLayer(Layer):
    """Couche custom pour calculer la distance euclidienne"""
    
    def __init__(self, **kwargs):
        super(EuclideanDistanceLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        x, y = inputs
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)
    
    def get_config(self):
        return super(EuclideanDistanceLayer, self).get_config()

class CosineSimilarityLayer(Layer):
    """Couche custom pour calculer la similarité cosinus"""
    
    def __init__(self, **kwargs):
        super(CosineSimilarityLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        x, y = inputs
        x_norm = K.l2_normalize(x, axis=-1)
        y_norm = K.l2_normalize(y, axis=-1)
        return K.sum(x_norm * y_norm, axis=-1, keepdims=True)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)
    
    def get_config(self):
        return super(CosineSimilarityLayer, self).get_config()

class ManhattanDistanceLayer(Layer):
    """Couche custom pour calculer la distance de Manhattan"""
    
    def __init__(self, **kwargs):
        super(ManhattanDistanceLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        x, y = inputs
        return K.sum(K.abs(x - y), axis=1, keepdims=True)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)
    
    def get_config(self):
        return super(ManhattanDistanceLayer, self).get_config()

# Chemins
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "siamese_entity_matcher.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "..", "models", "tokenizer.pkl")

print("📂 Chemin modèle :", MODEL_PATH)
print("📂 Chemin tokenizer :", TOKENIZER_PATH)

try:
    # Dictionnaire des objets custom pour le chargement
    custom_objects = {
        'EuclideanDistanceLayer': EuclideanDistanceLayer,
        'CosineSimilarityLayer': CosineSimilarityLayer,
        'ManhattanDistanceLayer': ManhattanDistanceLayer
    }
    
    # Charger le modèle avec les couches custom
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects=custom_objects
    )
    print("✅ Modèle chargé avec succès")
    print(f"📊 Architecture du modèle:")
    model.summary()

    # Charger le tokenizer
    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)
    print("✅ Tokenizer chargé avec succès")
    print(f"📊 Taille du vocabulaire: {len(tokenizer.word_index) + 1}")
    
    # Test simple de prédiction
    print("\n🧪 Test de prédiction simple...")
    
    # Préparer des données de test
    import numpy as np
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Textes de test
    text1 = "mounir lamsayah nador12a sb1113784 "
    text2 = "mohamed lamsayah nador12a sb11212"
    
    # Tokeniser et pad
    seq1 = tokenizer.texts_to_sequences([text1])
    seq2 = tokenizer.texts_to_sequences([text2])
    
    seq1_pad = pad_sequences(seq1, maxlen=200, padding='post')
    seq2_pad = pad_sequences(seq2, maxlen=200, padding='post')
    
    print(f"Forme des séquences: {seq1_pad.shape}, {seq2_pad.shape}")
    
    # Prédiction
    prediction = model.predict([seq1_pad, seq2_pad])
    similarity_score = float(prediction[0][0])
    
    print(f"✅ Prédiction réussie!")
    print(f"📈 Score de similarité: {similarity_score:.4f} ({similarity_score*100:.2f}%)")
    print(f"🎯 Correspondance: {'Oui' if similarity_score > 0.5 else 'Non'}")

except Exception as e:
    print(f"❌ Erreur : {e}")
    import traceback
    print("Stack trace complète:")
    traceback.print_exc()