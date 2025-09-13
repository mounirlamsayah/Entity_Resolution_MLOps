import pandas as pd
import numpy as np
import unicodedata
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.tokenizer = None
        
    def normalize(self, text):
        """Normalise le texte (enlève accents et caractères spéciaux)"""
        if pd.isnull(text):
            return ""
        text = str(text).lower()
        text = ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')
        text = re.sub(r'[^a-z0-9]', '', text)
        return text

    def clean_text(self, text):
        """Nettoie le texte pour la tokenisation"""
        if pd.isnull(text):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def load_data(self):
        """Charge les fichiers CSV"""
        print("Chargement des données...")
        reference_path = os.path.join(self.data_path, "reference_final.csv")
        source_path = os.path.join(self.data_path, "source_final.csv")
        
        reference = pd.read_csv(reference_path, delimiter=';')
        source = pd.read_csv(source_path, delimiter=';')
        
        print(f"Reference: {len(reference)} lignes")
        print(f"Source: {len(source)} lignes")
        
        return reference, source

    def create_positive_pairs(self, source, reference):
        """Crée les paires positives par merge exact"""
        print("Création des paires positives...")
        
        # Normalisation des clés
        source['key'] = source['nom_prenom_rs_clean'].apply(self.normalize)
        reference['key'] = reference['nom_prenom_rs_clean'].apply(self.normalize)
        
        # Merge exact
        positives_merged = source.merge(reference, on='key', suffixes=('_src', '_ref'))
        print(f"Paires positives initiales : {len(positives_merged)}")
        
        # Filtrage sur IFU, CIN ou ICE
        positive_pairs_filtered = positives_merged[
            (positives_merged['ifu_clean_src'] == positives_merged['ifu_clean_ref']) |
            (positives_merged['num_cin_clean_src'] == positives_merged['num_cin_clean_ref']) |
            (positives_merged['ice_clean_src'] == positives_merged['ice_clean_ref'])
        ]
        
        # Paires fiables (similarité nom >= 90% ET IFU match)
        from rapidfuzz import fuzz
        
        def similarity_score(row):
            return fuzz.ratio(str(row['nom_prenom_rs_clean_src']),
                            str(row['nom_prenom_rs_clean_ref']))
        
        positives_merged['similarity_nom'] = positives_merged.apply(similarity_score, axis=1)
        positives_merged['ifu_match'] = positives_merged.apply(
            lambda row: str(row['ifu_clean_src']) == str(row['ifu_clean_ref']), axis=1
        )
        
        trusted_pairs = positives_merged[
            (positives_merged['similarity_nom'] >= 90) &
            (positives_merged['ifu_match'] == True)
        ]
        
        trusted_pairs = trusted_pairs.drop(columns=['similarity_nom', 'ifu_match'])
        
        # Éliminer les doublons
        intersection = pd.merge(positive_pairs_filtered, trusted_pairs)
        trusted_pairs = trusted_pairs[~trusted_pairs.apply(tuple, 1).isin(intersection.apply(tuple, 1))]
        
        # Filtrer les lignes sans CIN ET sans adresse
        trusted_pairs = trusted_pairs[~trusted_pairs[['num_cin_clean_src', 'adresse_clean_src']].isna().all(axis=1)]
        
        # Combiner les paires
        positive_pairs = pd.concat([positive_pairs_filtered, trusted_pairs], ignore_index=True)
        positive_pairs = positive_pairs.sample(frac=1).reset_index(drop=True)
        
        print(f"Paires positives finales : {len(positive_pairs)}")
        return positive_pairs

    def create_negative_pairs(self, source, reference, positive_pairs):
        """Crée les paires négatives"""
        print("Création des paires négatives...")
        
        # Échantillonnage pour éviter la surcharge mémoire
        reference_sample = reference.sample(n=min(5000, len(reference)), random_state=42)
        source_sample = source.sample(n=min(1000, len(source)), random_state=42)
        
        # Produit cartésien
        fusion = source_sample.merge(reference_sample, how="cross", suffixes=('_src', '_ref'))
        
        # Supprimer les colonnes de clé
        fusion = fusion.drop(columns=['key_src', 'key_ref'], errors='ignore')
        positive_pairs_clean = positive_pairs.drop(columns=['key'], errors='ignore')
        
        # Éliminer les paires positives
        negatives = fusion.merge(
            positive_pairs_clean,
            on=list(fusion.columns),
            how="left",
            indicator=True
        )
        negatives = negatives[negatives["_merge"] == "left_only"].drop(columns="_merge")
        
        # Échantillonner les négatives
        n_negatives = len(positive_pairs_clean) + 50  # Légèrement plus pour équilibrer
        negative_pairs = negatives.sample(n=min(n_negatives, len(negatives)), random_state=42)
        
        print(f"Paires négatives créées : {len(negative_pairs)}")
        return negative_pairs

    def create_dataset(self, positive_pairs, negative_pairs):
        """Assemble le dataset final"""
        print("Assemblage du dataset final...")
        
        # Ajouter les labels
        positive_pairs["label"] = 1
        negative_pairs["label"] = 0
        
        # Combiner
        dataset = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Nettoyer les colonnes de texte
        columns_to_clean = [
            'nom_prenom_rs_clean_src', 'nom_prenom_rs_clean_ref',
            'adresse_clean_src', 'adresse_clean_ref',
            'num_cin_clean_src', 'num_cin_clean_ref'
        ]
        
        for col in columns_to_clean:
            dataset[col] = dataset[col].apply(self.clean_text)
        
        # Créer les inputs combinés
        dataset['input1'] = (dataset['nom_prenom_rs_clean_src'] + ' ' + 
                           dataset['adresse_clean_src'] + ' ' + 
                           dataset['num_cin_clean_src'])
        dataset['input2'] = (dataset['nom_prenom_rs_clean_ref'] + ' ' + 
                           dataset['adresse_clean_ref'] + ' ' + 
                           dataset['num_cin_clean_ref'])
        
        print(f"Dataset final : {len(dataset)} paires")
        print(f"Positives : {sum(dataset['label'])} | Négatives : {len(dataset) - sum(dataset['label'])}")
        
        return dataset

    def tokenize_and_pad(self, dataset, max_len=200):
        """Tokenise et pad les séquences"""
        print("Tokenisation et padding...")
        
        # Créer le tokenizer
        all_text = pd.concat([dataset['input1'], dataset['input2']])
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(all_text)
        
        vocab_size = len(self.tokenizer.word_index) + 1
        print(f"Taille du vocabulaire : {vocab_size}")
        
        # Convertir en séquences
        X1 = self.tokenizer.texts_to_sequences(dataset['input1'])
        X2 = self.tokenizer.texts_to_sequences(dataset['input2'])
        
        # Padding
        X1_pad = pad_sequences(X1, maxlen=max_len, padding='post')
        X2_pad = pad_sequences(X2, maxlen=max_len, padding='post')
        
        y = dataset['label'].values
        
        print(f"Forme X1_pad : {X1_pad.shape}")
        print(f"Forme X2_pad : {X2_pad.shape}")
        print(f"Forme y : {y.shape}")
        
        return X1_pad, X2_pad, y

    def save_data(self, X1_pad, X2_pad, y, dataset):
        """Sauvegarde les données préprocessées"""
        print("Sauvegarde des données...")
        
        os.makedirs("models", exist_ok=True)
        
        # Sauvegarder les arrays numpy
        np.save("models/X1_pad.npy", X1_pad)
        np.save("models/X2_pad.npy", X2_pad)
        np.save("models/y.npy", y)
        
        # Sauvegarder le tokenizer
        with open('models/tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Sauvegarder le dataset brut
        dataset.to_csv("models/processed_dataset.csv", index=False)
        
        print("Données sauvegardées dans le dossier models/")

    def split_data(self, X1_pad, X2_pad, y, test_size=0.2):
        """Divise les données en train/test"""
        return train_test_split(X1_pad, X2_pad, y, test_size=test_size, 
                              random_state=42, stratify=y)

def main():
    """Fonction principale"""
    preprocessor = DataPreprocessor()
    
    # Charger les données
    reference, source = preprocessor.load_data()
    
    # Créer les paires positives
    positive_pairs = preprocessor.create_positive_pairs(source, reference)
    
    # Créer les paires négatives
    negative_pairs = preprocessor.create_negative_pairs(source, reference, positive_pairs)
    
    # Créer le dataset
    dataset = preprocessor.create_dataset(positive_pairs, negative_pairs)
    
    # Tokeniser et padder
    X1_pad, X2_pad, y = preprocessor.tokenize_and_pad(dataset)
    
    # Sauvegarder
    preprocessor.save_data(X1_pad, X2_pad, y, dataset)
    
    # Split train/test et sauvegarder
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = preprocessor.split_data(X1_pad, X2_pad, y)
    
    np.save("models/X1_train.npy", X1_train)
    np.save("models/X1_test.npy", X1_test)
    np.save("models/X2_train.npy", X2_train)
    np.save("models/X2_test.npy", X2_test)
    np.save("models/y_train.npy", y_train)
    np.save("models/y_test.npy", y_test)
    
    print("\n✅ Préprocessing terminé avec succès!")
    print(f"📊 Train: {len(X1_train)} échantillons")
    print(f"📊 Test: {len(X1_test)} échantillons")

if __name__ == "__main__":
    main()