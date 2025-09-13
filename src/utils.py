import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

def check_data_files():
    """Vérifie la présence des fichiers de données requis"""
    required_files = [
        "data/source_final.csv",
        "data/reference_final.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Fichiers manquants:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✅ Tous les fichiers de données sont présents")
        return True

def create_directories():
    """Crée les répertoires nécessaires"""
    directories = ["models", "data", "logs", "outputs"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Répertoire créé/vérifié: {directory}")

def log_experiment(experiment_name, metrics, config=None):
    """Enregistre les résultats d'une expérience"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_data = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "metrics": metrics
    }
    
    if config:
        experiment_data["config"] = config
    
    # Créer le fichier de log
    log_file = f"logs/experiment_{experiment_name}_{timestamp}.json"
    
    with open(log_file, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"📝 Expérience enregistrée: {log_file}")

def get_dataset_info():
    """Retourne des informations sur le dataset"""
    try:
        if os.path.exists("models/processed_dataset.csv"):
            df = pd.read_csv("models/processed_dataset.csv")
            
            info = {
                "total_samples": len(df),
                "positive_samples": sum(df['label']),
                "negative_samples": len(df) - sum(df['label']),
                "balance_ratio": sum(df['label']) / len(df),
                "columns": list(df.columns)
            }
            
            print("📊 Informations du dataset:")
            print(f"  - Total: {info['total_samples']} échantillons")
            print(f"  - Positifs: {info['positive_samples']} ({info['balance_ratio']:.2%})")
            print(f"  - Négatifs: {info['negative_samples']} ({1-info['balance_ratio']:.2%})")
            
            return info
        else:
            print("❌ Dataset traité non trouvé")
            return None
            
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du dataset: {e}")
        return None

def validate_preprocessing_outputs():
    """Valide les sorties du preprocessing"""
    required_files = [
        "models/X1_train.npy",
        "models/X2_train.npy", 
        "models/y_train.npy",
        "models/X1_test.npy",
        "models/X2_test.npy",
        "models/y_test.npy",
        "models/tokenizer.pkl",
        "models/processed_dataset.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Fichiers de preprocessing manquants:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✅ Tous les fichiers de preprocessing sont présents")
        
        # Vérifier la cohérence des dimensions
        try:
            X1_train = np.load("models/X1_train.npy")
            X2_train = np.load("models/X2_train.npy")
            y_train = np.load("models/y_train.npy")
            
            print(f"📊 Dimensions d'entraînement:")
            print(f"  - X1_train: {X1_train.shape}")
            print(f"  - X2_train: {X2_train.shape}")
            print(f"  - y_train: {y_train.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la validation: {e}")
            return False

def validate_model_outputs():
    """Valide les sorties de l'entraînement"""
    model_files = [
        "models/siamese_entity_matcher.keras",
        "models/training_metrics.json"
    ]
    
    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Fichiers de modèle manquants:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✅ Tous les fichiers de modèle sont présents")
        
        # Lire les métriques d'entraînement
        try:
            with open("models/training_metrics.json", 'r') as f:
                metrics = json.load(f)
            
            print("📊 Métriques d'entraînement finales:")
            for key, value in metrics.items():
                print(f"  - {key}: {value:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la lecture des métriques: {e}")
            return False

def cleanup_temp_files():
    """Nettoie les fichiers temporaires"""
    temp_patterns = [
        "*.tmp",
        "*.log",
        "checkpoint*",
        "__pycache__",
    ]
    
    cleaned_files = 0
    for pattern in temp_patterns:
        import glob
        files = glob.glob(pattern, recursive=True)
        for file in files:
            try:
                if os.path.isfile(file):
                    os.remove(file)
                    cleaned_files += 1
                elif os.path.isdir(file):
                    import shutil
                    shutil.rmtree(file)
                    cleaned_files += 1
            except Exception as e:
                print(f"⚠️ Impossible de supprimer {file}: {e}")
    
    if cleaned_files > 0:
        print(f"🧹 {cleaned_files} fichiers temporaires nettoyés")
    else:
        print("✅ Aucun fichier temporaire à nettoyer")

def generate_project_report():
    """Génère un rapport complet du projet"""
    report = {
        "project_status": {},
        "dataset_info": {},
        "model_info": {},
        "generated_at": datetime.now().isoformat()
    }
    
    # Statut des fichiers
    report["project_status"]["data_files_ok"] = check_data_files()
    report["project_status"]["preprocessing_ok"] = validate_preprocessing_outputs()
    report["project_status"]["model_ok"] = validate_model_outputs()
    
    # Informations dataset
    dataset_info = get_dataset_info()
    if dataset_info:
        report["dataset_info"] = dataset_info
    
    # Informations modèle
    if os.path.exists("models/training_metrics.json"):
        with open("models/training_metrics.json", 'r') as f:
            report["model_info"]["training_metrics"] = json.load(f)
    
    if os.path.exists("models/evaluation_results.json"):
        with open("models/evaluation_results.json", 'r') as f:
            report["model_info"]["evaluation_results"] = json.load(f)
    
    # Sauvegarder le rapport
    report_file = f"outputs/project_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📋 Rapport projet généré: {report_file}")
    
    # Affichage résumé
    print("\n" + "="*50)
    print("📊 RÉSUMÉ DU PROJET")
    print("="*50)
    print(f"✅ Fichiers de données: {'OK' if report['project_status']['data_files_ok'] else 'MANQUANT'}")
    print(f"✅ Preprocessing: {'OK' if report['project_status']['preprocessing_ok'] else 'MANQUANT'}")
    print(f"✅ Modèle: {'OK' if report['project_status']['model_ok'] else 'MANQUANT'}")
    
    if dataset_info:
        print(f"📊 Dataset: {dataset_info['total_samples']} échantillons")
    
    return report

def check_dependencies():
    """Vérifie les dépendances Python"""
    required_packages = [
        'pandas', 'numpy', 'tensorflow', 'scikit-learn', 
        'rapidfuzz', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Packages Python manquants:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstallez avec: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ Toutes les dépendances Python sont installées")
        return True

def main():
    """Fonction principale des utilitaires"""
    print("🛠️ Utilitaires du projet Entity Matching")
    print("="*50)
    
    # Créer les répertoires
    create_directories()
    
    # Vérifier les dépendances
    check_dependencies()
    
    # Générer le rapport
    generate_project_report()
    
    print("\n✅ Vérifications terminées!")

if __name__ == "__main__":
    main()

    