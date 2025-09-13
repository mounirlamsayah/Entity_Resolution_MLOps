import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, LSTM, Dense, Lambda, 
                                   Dropout, BatchNormalization, Bidirectional, 
                                   GlobalMaxPooling1D, concatenate, Layer)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class SiameseConfig:
    def __init__(self):
        self.vocab_size = None
        self.embedding_dim = 128
        self.max_len = 200
        self.lstm_units = 64
        self.dropout_rate = 0.3
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 10

# ===== DÉFINITION DES COUCHES CUSTOM =====
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

class SiameseTrainer:
    def __init__(self, config=None):
        self.config = config if config else SiameseConfig()
        self.model = None
        self.history = None
        
    def load_data(self):
        """Charge les données préprocessées"""
        print("Chargement des données d'entraînement...")
        
        X1_train = np.load("models/X1_train.npy")
        X2_train = np.load("models/X2_train.npy")
        y_train = np.load("models/y_train.npy")
        X1_test = np.load("models/X1_test.npy")
        X2_test = np.load("models/X2_test.npy")
        y_test = np.load("models/y_test.npy")
        
        with open('models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        self.config.vocab_size = len(tokenizer.word_index) + 1
        
        print(f"Train: {len(X1_train)} échantillons")
        print(f"Test: {len(X1_test)} échantillons")
        print(f"Vocabulaire: {self.config.vocab_size}")
        
        return X1_train, X2_train, y_train, X1_test, X2_test, y_test, tokenizer

    def create_siamese_model(self):
        """Crée le modèle Siamese Neural Network avec couches custom"""
        print("Construction du modèle Siamese...")
        
        # Entrées
        input_a = Input(shape=(self.config.max_len,), name='input_a')
        input_b = Input(shape=(self.config.max_len,), name='input_b')
        
        # Embedding partagé
        embedding_layer = Embedding(
            input_dim=self.config.vocab_size,
            output_dim=self.config.embedding_dim,
            input_length=self.config.max_len,
            mask_zero=True,
            name='shared_embedding'
        )
        
        # Architecture de l'encodeur partagé
        def create_encoder():
            lstm = Bidirectional(
                LSTM(self.config.lstm_units, return_sequences=True, dropout=0.2),
                name='bidirectional_lstm'
            )
            pooling = GlobalMaxPooling1D(name='global_max_pooling')
            dense1 = Dense(128, activation='relu', name='dense_1')
            bn1 = BatchNormalization(name='batch_norm_1')
            dropout1 = Dropout(self.config.dropout_rate, name='dropout_1')
            dense2 = Dense(64, activation='relu', name='dense_2')
            bn2 = BatchNormalization(name='batch_norm_2')
            dropout2 = Dropout(self.config.dropout_rate, name='dropout_2')
            
            def encoder(x):
                x = lstm(x)
                x = pooling(x)
                x = dense1(x)
                x = bn1(x)
                x = dropout1(x)
                x = dense2(x)
                x = bn2(x)
                x = dropout2(x)
                return x
            return encoder
        
        encoder = create_encoder()
        
        # Appliquer l'embedding et l'encodeur
        embedded_a = embedding_layer(input_a)
        embedded_b = embedding_layer(input_b)
        encoded_a = encoder(embedded_a)
        encoded_b = encoder(embedded_b)
        
        # Utiliser les couches custom au lieu des fonctions Lambda
        euclidean_dist = EuclideanDistanceLayer(name='euclidean_distance')([encoded_a, encoded_b])
        cosine_sim = CosineSimilarityLayer(name='cosine_similarity')([encoded_a, encoded_b])
        manhattan_dist = ManhattanDistanceLayer(name='manhattan_distance')([encoded_a, encoded_b])
        
        # Combiner les métriques
        combined_features = concatenate([euclidean_dist, cosine_sim, manhattan_dist], 
                                      name='combined_features')
        
        # Couches de classification
        dense_final_1 = Dense(32, activation='relu', name='final_dense_1')(combined_features)
        dropout_final_1 = Dropout(0.2, name='final_dropout_1')(dense_final_1)
        dense_final_2 = Dense(16, activation='relu', name='final_dense_2')(dropout_final_1)
        dropout_final_2 = Dropout(0.2, name='final_dropout_2')(dense_final_2)
        output = Dense(1, activation='sigmoid', name='output')(dropout_final_2)
        
        # Créer le modèle
        model = Model(inputs=[input_a, input_b], outputs=output, name='SiameseEntityMatcher')
        
        return model

    def compile_model(self, model):
        """Compile le modèle"""
        optimizer = Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        return model

    def create_callbacks(self):
        """Crée les callbacks d'entraînement"""
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        return [early_stopping, reduce_lr, checkpoint]

    def train_model(self, X1_train, X2_train, y_train, X1_test, X2_test, y_test):
        """Entraîne le modèle"""
        print("Début de l'entraînement...")
        
        # Créer et compiler le modèle
        self.model = self.create_siamese_model()
        self.model = self.compile_model(self.model)
        
        # Afficher l'architecture
        self.model.summary()
        
        # Callbacks
        callbacks = self.create_callbacks()
        
        # Entraînement
        self.history = self.model.fit(
            [X1_train, X2_train], y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Évaluation sur le test
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            [X1_test, X2_test], y_test, verbose=0
        )
        
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        print(f"\n📊 Résultats sur le jeu de test:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        
        # Prédictions détaillées
        y_pred_proba = self.model.predict([X1_test, X2_test])
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        print("\n📋 Rapport de classification:")
        print(classification_report(y_test, y_pred))
        
        print("\n📢 Matrice de confusion:")
        print(confusion_matrix(y_test, y_pred))
        
        return self.model, self.history

    def save_model(self, model_path="models/siamese_entity_matcher.h5"):
        """Sauvegarde le modèle final"""
        print(f"Sauvegarde du modèle : {model_path}")
        self.model.save(model_path)
        
        # Sauvegarder aussi les métriques
        metrics = {
            'final_train_accuracy': self.history.history['accuracy'][-1],
            'final_val_accuracy': self.history.history['val_accuracy'][-1],
            'final_train_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1],
        }
        
        import json
        with open('models/training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("✅ Modèle et métriques sauvegardés!")

    def plot_training_history(self):
        """Visualise l'historique d'entraînement"""
        if self.history is None:
            print("Aucun historique d'entraînement disponible")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Val Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy
        ax2.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        # Precision
        ax3.plot(self.history.history['precision'], label='Train Precision')
        ax3.plot(self.history.history['val_precision'], label='Val Precision')
        ax3.set_title('Model Precision')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Precision')
        ax3.legend()
        
        # Recall
        ax4.plot(self.history.history['recall'], label='Train Recall')
        ax4.plot(self.history.history['val_recall'], label='Val Recall')
        ax4.set_title('Model Recall')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Recall')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Fonction principale d'entraînement"""
    print("🚀 Début de l'entraînement du modèle Siamese")
    
    # Configuration
    config = SiameseConfig()
    trainer = SiameseTrainer(config)
    
    # Charger les données
    X1_train, X2_train, y_train, X1_test, X2_test, y_test, tokenizer = trainer.load_data()
    
    # Entraîner
    model, history = trainer.train_model(X1_train, X2_train, y_train, 
                                        X1_test, X2_test, y_test)
    
    # Sauvegarder
    trainer.save_model()
    
    # Visualiser
    trainer.plot_training_history()
    
    print("\n✅ Entraînement terminé avec succès!")
    print("📁 Fichiers générés dans models/:")
    print("  - siamese_entity_matcher.h5 (modèle final)")
    print("  - best_model.h5 (meilleur modèle)")
    print("  - training_metrics.json (métriques)")
    print("  - training_history.png (graphiques)")

if __name__ == "__main__":
    main()