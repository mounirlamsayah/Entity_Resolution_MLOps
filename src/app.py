import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import pickle
import re
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import json
from datetime import datetime

# ===== DÉFINITION DES COUCHES CUSTOM POUR LE CHARGEMENT =====
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

class EntityMatcher:
    def __init__(self, model_path="models/siamese_entity_matcher.h5", 
                 tokenizer_path="models/tokenizer.pkl"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None
        self.max_len = 200
        
    def load_model_and_tokenizer(self):
        """Charge le modèle et le tokenizer"""
        print("Chargement du modèle et tokenizer...")
        
        try:
            # Charger le modèle avec les couches custom
            custom_objects = {
                'EuclideanDistanceLayer': EuclideanDistanceLayer,
                'CosineSimilarityLayer': CosineSimilarityLayer,
                'ManhattanDistanceLayer': ManhattanDistanceLayer
            }
            
            self.model = load_model(self.model_path, custom_objects=custom_objects)
            print(f"✅ Modèle chargé depuis {self.model_path}")
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            return False
            
        try:
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"✅ Tokenizer chargé depuis {self.tokenizer_path}")
        except Exception as e:
            print(f"❌ Erreur chargement tokenizer: {e}")
            return False
            
        return True
    
    def clean_text(self, text):
        """Nettoie le texte comme lors du preprocessing"""
        if pd.isnull(text) or text == "":
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9 ]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def prepare_text_pair(self, text1, text2):
        """Prépare une paire de textes pour la prédiction"""
        # Nettoyer les textes
        text1_clean = self.clean_text(text1)
        text2_clean = self.clean_text(text2)
        
        # Tokeniser
        seq1 = self.tokenizer.texts_to_sequences([text1_clean])
        seq2 = self.tokenizer.texts_to_sequences([text2_clean])
        
        # Padding
        seq1_pad = pad_sequences(seq1, maxlen=self.max_len, padding='post')
        seq2_pad = pad_sequences(seq2, maxlen=self.max_len, padding='post')
        
        return seq1_pad, seq2_pad
    
    def predict_similarity(self, text1, text2):
        """Prédit la similarité entre deux textes"""
        if self.model is None or self.tokenizer is None:
            print("❌ Modèle ou tokenizer non chargé")
            return None
            
        try:
            # Préparer les données
            seq1_pad, seq2_pad = self.prepare_text_pair(text1, text2)
            
            # Prédiction
            similarity_score = self.model.predict([seq1_pad, seq2_pad], verbose=0)[0][0]
            
            return float(similarity_score)
            
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction: {e}")
            return None
    
    def predict_batch(self, text_pairs):
        """Prédit pour un batch de paires de textes"""
        if self.model is None or self.tokenizer is None:
            print("❌ Modèle ou tokenizer non chargé")
            return None
        
        try:
            results = []
            for i, (text1, text2) in enumerate(text_pairs):
                similarity = self.predict_similarity(text1, text2)
                results.append({
                    'pair_id': i,
                    'text1': text1,
                    'text2': text2,
                    'similarity_score': similarity,
                    'is_match': similarity > 0.5 if similarity is not None else False
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction batch: {e}")
            return None
    
    def match_entity_to_database(self, query_entity, database_entities, threshold=0.5):
        """Trouve les meilleures correspondances pour une entité dans une base de données"""
        if not database_entities:
            return []
        
        matches = []
        for i, db_entity in enumerate(database_entities):
            similarity = self.predict_similarity(query_entity, db_entity)
            if similarity is not None and similarity >= threshold:
                matches.append({
                    'entity_id': i,
                    'entity_text': db_entity,
                    'similarity_score': similarity,
                    'confidence': 'High' if similarity > 0.8 else 'Medium' if similarity > 0.65 else 'Low'
                })
        
        # Trier par score de similarité décroissant
        matches = sorted(matches, key=lambda x: x['similarity_score'], reverse=True)
        return matches
    
    def evaluate_on_test_data(self, test_data_path="models/processed_dataset.csv"):
        """Évalue le modèle sur des données de test"""
        try:
            # Charger les données de test
            test_data = pd.read_csv(test_data_path)
            
            # Préparer les données
            predictions = []
            true_labels = []
            
            print("Évaluation en cours...")
            for idx, row in test_data.iterrows():
                text1 = f"{row.get('nom_prenom_rs_clean_src', '')} {row.get('adresse_clean_src', '')} {row.get('num_cin_clean_src', '')}"
                text2 = f"{row.get('nom_prenom_rs_clean_ref', '')} {row.get('adresse_clean_ref', '')} {row.get('num_cin_clean_ref', '')}"
                
                similarity = self.predict_similarity(text1, text2)
                if similarity is not None:
                    predictions.append(1 if similarity > 0.5 else 0)
                    true_labels.append(row['label'])
                
                if (idx + 1) % 100 == 0:
                    print(f"Traité {idx + 1}/{len(test_data)} échantillons")
            
            # Calculer les métriques
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions)
            recall = recall_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions)
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'total_samples': len(predictions)
            }
            
            return metrics
            
        except Exception as e:
            print(f"❌ Erreur lors de l'évaluation: {e}")
            return None

# Application Flask
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Instance globale du matcher
matcher = None

def initialize_matcher():
    """Initialise le matcher au démarrage de l'application"""
    global matcher
    matcher = EntityMatcher()
    success = matcher.load_model_and_tokenizer()
    if not success:
        print("⚠️ Échec du chargement du modèle. Certaines fonctionnalités seront indisponibles.")
    return success

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API pour prédire la similarité entre deux textes"""
    try:
        data = request.get_json()
        
        if not data or 'text1' not in data or 'text2' not in data:
            return jsonify({'error': 'Les champs text1 et text2 sont requis'}), 400
        
        text1 = data['text1']
        text2 = data['text2']
        
        if matcher is None:
            return jsonify({'error': 'Modèle non initialisé'}), 500
        
        similarity = matcher.predict_similarity(text1, text2)
        
        if similarity is None:
            return jsonify({'error': 'Erreur lors de la prédiction'}), 500
        
        result = {
            'text1': text1,
            'text2': text2,
            'similarity_score': similarity,
            'is_match': similarity > 0.5,
            'confidence': 'High' if similarity > 0.8 else 'Medium' if similarity > 0.65 else 'Low',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/api/batch_predict', methods=['POST'])
def api_batch_predict():
    """API pour prédictions en lot"""
    try:
        data = request.get_json()
        
        if not data or 'pairs' not in data:
            return jsonify({'error': 'Le champ pairs est requis'}), 400
        
        pairs = data['pairs']
        
        if not isinstance(pairs, list) or len(pairs) == 0:
            return jsonify({'error': 'pairs doit être une liste non vide'}), 400
        
        # Valider le format des paires
        text_pairs = []
        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict) or 'text1' not in pair or 'text2' not in pair:
                return jsonify({'error': f'Paire {i} invalide: text1 et text2 requis'}), 400
            text_pairs.append((pair['text1'], pair['text2']))
        
        if matcher is None:
            return jsonify({'error': 'Modèle non initialisé'}), 500
        
        results = matcher.predict_batch(text_pairs)
        
        if results is None:
            return jsonify({'error': 'Erreur lors de la prédiction batch'}), 500
        
        response = {
            'results': results,
            'total_pairs': len(results),
            'matches_found': sum(1 for r in results if r['is_match']),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/api/match_entity', methods=['POST'])
def api_match_entity():
    """API pour trouver des correspondances dans une base de données"""
    try:
        data = request.get_json()
        
        if not data or 'query_entity' not in data or 'database_entities' not in data:
            return jsonify({'error': 'query_entity et database_entities sont requis'}), 400
        
        query_entity = data['query_entity']
        database_entities = data['database_entities']
        threshold = data.get('threshold', 0.5)
        
        if not isinstance(database_entities, list):
            return jsonify({'error': 'database_entities doit être une liste'}), 400
        
        if matcher is None:
            return jsonify({'error': 'Modèle non initialisé'}), 500
        
        matches = matcher.match_entity_to_database(query_entity, database_entities, threshold)
        
        response = {
            'query_entity': query_entity,
            'matches': matches,
            'total_matches': len(matches),
            'threshold_used': threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """API pour évaluer le modèle"""
    try:
        data = request.get_json()
        test_data_path = data.get('test_data_path', 'models/processed_dataset.csv') if data else 'models/processed_dataset.csv'
        
        if matcher is None:
            return jsonify({'error': 'Modèle non initialisé'}), 500
        
        metrics = matcher.evaluate_on_test_data(test_data_path)
        
        if metrics is None:
            return jsonify({'error': 'Erreur lors de l\'évaluation'}), 500
        
        response = {
            'evaluation_metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Vérification de l'état de l'application"""
    model_loaded = matcher is not None and matcher.model is not None
    tokenizer_loaded = matcher is not None and matcher.tokenizer is not None
    
    return jsonify({
        'status': 'healthy' if model_loaded and tokenizer_loaded else 'degraded',
        'model_loaded': model_loaded,
        'tokenizer_loaded': tokenizer_loaded,
        'timestamp': datetime.now().isoformat()
    })

# Template HTML pour l'interface web
html_template = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Entity Matcher - Correspondance d'Entités</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        textarea, input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        textarea {
            height: 100px;
            resize: vertical;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-right: 10px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
            background-color: #f8f9fa;
        }
        .score-high { color: #28a745; font-weight: bold; }
        .score-medium { color: #ffc107; font-weight: bold; }
        .score-low { color: #dc3545; font-weight: bold; }
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            background-color: #e9ecef;
            border: none;
            cursor: pointer;
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
        }
        .tab.active {
            background-color: #007bff;
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Entity Matcher</h1>
        <p style="text-align: center; color: #666;">Système de correspondance d'entités utilisant l'intelligence artificielle</p>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('simple')">Comparaison Simple</button>
            <button class="tab" onclick="showTab('batch')">Traitement par Lot</button>
            <button class="tab" onclick="showTab('search')">Recherche en Base</button>
        </div>
        
        <!-- Comparaison Simple -->
        <div id="simple" class="tab-content active">
            <form id="compareForm">
                <div class="form-group">
                    <label for="text1">Première entité :</label>
                    <textarea id="text1" name="text1" placeholder="Ex: Jean Dupont 123 rue de la Paix Paris CIN123456"></textarea>
                </div>
                <div class="form-group">
                    <label for="text2">Deuxième entité :</label>
                    <textarea id="text2" name="text2" placeholder="Ex: DUPONT Jean 123 RUE PAIX PARIS CIN123456"></textarea>
                </div>
                <button type="submit">Comparer</button>
            </form>
            <div id="result"></div>
        </div>
        
        <!-- Traitement par Lot -->
        <div id="batch" class="tab-content">
            <div class="form-group">
                <label for="batchInput">Paires à comparer (une par ligne, séparées par |) :</label>
                <textarea id="batchInput" placeholder="Jean Dupont|DUPONT Jean&#10;Marie Martin 75001|MARTIN Marie Paris"></textarea>
            </div>
            <button onclick="processBatch()">Traiter le Lot</button>
            <div id="batchResult"></div>
        </div>
        
        <!-- Recherche en Base -->
        <div id="search" class="tab-content">
            <div class="form-group">
                <label for="queryEntity">Entité à rechercher :</label>
                <textarea id="queryEntity" placeholder="Ex: Jean Dupont Paris CIN123456"></textarea>
            </div>
            <div class="form-group">
                <label for="database">Base de données (une entité par ligne) :</label>
                <textarea id="database" placeholder="DUPONT Jean 123 rue Paris&#10;Pierre Martin Lyon&#10;Marie Dubois Nice"></textarea>
            </div>
            <div class="form-group">
                <label for="threshold">Seuil de similarité :</label>
                <input type="number" id="threshold" value="0.5" min="0" max="1" step="0.1">
            </div>
            <button onclick="searchEntity()">Rechercher</button>
            <div id="searchResult"></div>
        </div>
    </div>

    <script>
        function showTab(tabName) {
            // Cacher tous les contenus d'onglets
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Désactiver tous les onglets
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Activer l'onglet et son contenu
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }

        // Comparaison simple
        document.getElementById('compareForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const text1 = document.getElementById('text1').value;
            const text2 = document.getElementById('text2').value;
            
            if (!text1 || !text2) {
                alert('Veuillez remplir les deux champs');
                return;
            }
            
            fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text1: text1,
                    text2: text2
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('result').innerHTML = '<div class="result" style="border-color: #dc3545;"><strong>Erreur:</strong> ' + data.error + '</div>';
                } else {
                    const scoreClass = data.similarity_score > 0.8 ? 'score-high' : 
                                      data.similarity_score > 0.65 ? 'score-medium' : 'score-low';
                    
                    document.getElementById('result').innerHTML = `
                        <div class="result">
                            <h3>Résultat de la comparaison</h3>
                            <p><strong>Score de similarité:</strong> <span class="${scoreClass}">${(data.similarity_score * 100).toFixed(2)}%</span></p>
                            <p><strong>Correspondance:</strong> ${data.is_match ? '✅ Oui' : '❌ Non'}</p>
                            <p><strong>Niveau de confiance:</strong> ${data.confidence}</p>
                        </div>
                    `;
                }
            })
            .catch(error => {
                document.getElementById('result').innerHTML = '<div class="result" style="border-color: #dc3545;"><strong>Erreur:</strong> ' + error + '</div>';
            });
        });

        // Traitement par lot
        function processBatch() {
            const batchInput = document.getElementById('batchInput').value;
            
            if (!batchInput) {
                alert('Veuillez entrer des paires à comparer');
                return;
            }
            
            const lines = batchInput.split('\\n').filter(line => line.trim());
            const pairs = lines.map(line => {
                const parts = line.split('|');
                if (parts.length !== 2) {
                    throw new Error('Format invalide: ' + line);
                }
                return {
                    text1: parts[0].trim(),
                    text2: parts[1].trim()
                };
            });
            
            fetch('/api/batch_predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    pairs: pairs
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('batchResult').innerHTML = '<div class="result" style="border-color: #dc3545;"><strong>Erreur:</strong> ' + data.error + '</div>';
                } else {
                    let html = `
                        <div class="result">
                            <h3>Résultats du traitement par lot</h3>
                            <p><strong>Paires traitées:</strong> ${data.total_pairs}</p>
                            <p><strong>Correspondances trouvées:</strong> ${data.matches_found}</p>
                            <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                                <tr style="background-color: #f8f9fa;">
                                    <th style="border: 1px solid #dee2e6; padding: 8px;">#</th>
                                    <th style="border: 1px solid #dee2e6; padding: 8px;">Texte 1</th>
                                    <th style="border: 1px solid #dee2e6; padding: 8px;">Texte 2</th>
                                    <th style="border: 1px solid #dee2e6; padding: 8px;">Score</th>
                                    <th style="border: 1px solid #dee2e6; padding: 8px;">Match</th>
                                </tr>
                    `;
                    
                    data.results.forEach((result, index) => {
                        const scoreClass = result.similarity_score > 0.8 ? 'score-high' : 
                                          result.similarity_score > 0.65 ? 'score-medium' : 'score-low';
                        html += `
                            <tr>
                                <td style="border: 1px solid #dee2e6; padding: 8px;">${index + 1}</td>
                                <td style="border: 1px solid #dee2e6; padding: 8px;">${result.text1}</td>
                                <td style="border: 1px solid #dee2e6; padding: 8px;">${result.text2}</td>
                                <td style="border: 1px solid #dee2e6; padding: 8px;"><span class="${scoreClass}">${(result.similarity_score * 100).toFixed(1)}%</span></td>
                                <td style="border: 1px solid #dee2e6; padding: 8px;">${result.is_match ? '✅' : '❌'}</td>
                            </tr>
                        `;
                    });
                    
                    html += '</table></div>';
                    document.getElementById('batchResult').innerHTML = html;
                }
            })
            .catch(error => {
                document.getElementById('batchResult').innerHTML = '<div class="result" style="border-color: #dc3545;"><strong>Erreur:</strong> ' + error + '</div>';
            });
        }

        // Recherche en base
        function searchEntity() {
            const queryEntity = document.getElementById('queryEntity').value;
            const database = document.getElementById('database').value;
            const threshold = parseFloat(document.getElementById('threshold').value);
            
            if (!queryEntity || !database) {
                alert('Veuillez remplir tous les champs');
                return;
            }
            
            const databaseEntities = database.split('\\n').filter(line => line.trim()).map(line => line.trim());
            
            fetch('/api/match_entity', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query_entity: queryEntity,
                    database_entities: databaseEntities,
                    threshold: threshold
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('searchResult').innerHTML = '<div class="result" style="border-color: #dc3545;"><strong>Erreur:</strong> ' + data.error + '</div>';
                } else {
                    let html = `
                        <div class="result">
                            <h3>Résultats de la recherche</h3>
                            <p><strong>Entité recherchée:</strong> ${data.query_entity}</p>
                            <p><strong>Correspondances trouvées:</strong> ${data.total_matches}</p>
                            <p><strong>Seuil utilisé:</strong> ${(data.threshold_used * 100).toFixed(0)}%</p>
                    `;
                    
                    if (data.matches.length > 0) {
                        html += `
                            <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                                <tr style="background-color: #f8f9fa;">
                                    <th style="border: 1px solid #dee2e6; padding: 8px;">#</th>
                                    <th style="border: 1px solid #dee2e6; padding: 8px;">Entité correspondante</th>
                                    <th style="border: 1px solid #dee2e6; padding: 8px;">Score</th>
                                    <th style="border: 1px solid #dee2e6; padding: 8px;">Confiance</th>
                                </tr>
                        `;
                        
                        data.matches.forEach((match, index) => {
                            const scoreClass = match.similarity_score > 0.8 ? 'score-high' : 
                                              match.similarity_score > 0.65 ? 'score-medium' : 'score-low';
                            html += `
                                <tr>
                                    <td style="border: 1px solid #dee2e6; padding: 8px;">${index + 1}</td>
                                    <td style="border: 1px solid #dee2e6; padding: 8px;">${match.entity_text}</td>
                                    <td style="border: 1px solid #dee2e6; padding: 8px;"><span class="${scoreClass}">${(match.similarity_score * 100).toFixed(1)}%</span></td>
                                    <td style="border: 1px solid #dee2e6; padding: 8px;">${match.confidence}</td>
                                </tr>
                            `;
                        });
                        
                        html += '</table>';
                    } else {
                        html += '<p style="color: #666; font-style: italic; margin-top: 15px;">Aucune correspondance trouvée avec le seuil spécifié.</p>';
                    }
                    
                    html += '</div>';
                    document.getElementById('searchResult').innerHTML = html;
                }
            })
            .catch(error => {
                document.getElementById('searchResult').innerHTML = '<div class="result" style="border-color: #dc3545;"><strong>Erreur:</strong> ' + error + '</div>';
            });
        }
    </script>
</body>
</html>
'''

# Créer le dossier templates s'il n'existe pas
os.makedirs('templates', exist_ok=True)

# Sauvegarder le template HTML
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

if __name__ == '__main__':
    print("🚀 Démarrage de l'application Entity Matcher")
    print("=" * 50)
    
    # Vérifier la présence des fichiers nécessaires
    model_path = "models/siamese_entity_matcher.h5"
    tokenizer_path = "models/tokenizer.pkl"
    
    if not os.path.exists(model_path):
        print(f"⚠️  Fichier modèle non trouvé: {model_path}")
        print("   Assurez-vous d'avoir exécuté l'entraînement du modèle d'abord.")
    
    if not os.path.exists(tokenizer_path):
        print(f"⚠️  Fichier tokenizer non trouvé: {tokenizer_path}")
        print("   Assurez-vous d'avoir exécuté le preprocessing d'abord.")
    
    # Initialiser le matcher
    success = initialize_matcher()
    
    if success:
        print("✅ Modèle et tokenizer chargés avec succès!")
        print("\n📊 Informations du modèle:")
        print(f"   - Longueur maximale des séquences: {matcher.max_len}")
        print(f"   - Taille du vocabulaire: {len(matcher.tokenizer.word_index) + 1}")
    else:
        print("⚠️  Certaines fonctionnalités seront limitées.")
    
    print("\n🌐 Interface web disponible sur:")
    print("   - http://localhost:5000")
    print("   - http://127.0.0.1:5000")
    
    print("\n🔗 Endpoints API disponibles:")
    print("   - POST /api/predict - Comparaison simple")
    print("   - POST /api/batch_predict - Traitement par lot")
    print("   - POST /api/match_entity - Recherche en base")
    print("   - POST /api/evaluate - Évaluation du modèle")
    print("   - GET /health - Vérification de l'état")
    
    print("\n" + "=" * 50)
    print("Appuyez sur Ctrl+C pour arrêter le serveur")
    
    try:
        # Démarrer l'application Flask
        app.run(
            debug=False,  # Mettre à True pour le développement
            host='0.0.0.0',  # Accessible depuis n'importe quelle IP
            port=5000,
            threaded=True  # Support des requêtes multiples
        )
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt du serveur. Au revoir!")
    except Exception as e:
        print(f"\n❌ Erreur lors du démarrage: {e}")