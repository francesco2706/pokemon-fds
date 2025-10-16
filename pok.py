# Pokemon Battles Prediction - Competition Grade Model (FIXED)
import json
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# --- Configurazione ---
DATA_PATH = r"C:\Users\giann\Desktop\universita\magistrale\FUNDATIONS OF DATA SCIENCE\progetto pokemon"
train_file_path = os.path.join(DATA_PATH, 'train.jsonl')
test_file_path = os.path.join(DATA_PATH, 'test.jsonl')

print("=" * 70)
print("POKEMON BATTLES PREDICTION - COMPETITION GRADE MODEL")
print("=" * 70)

# --------------------------------------------------------
# 1. Caricamento Dati Avanzato
# --------------------------------------------------------
def load_battles_with_progress(file_path):
    """Carica le battaglie con progress bar"""
    battles = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading battles"):
            battles.append(json.loads(line))
    return battles

print("\n1. Loading data...")
train_data = load_battles_with_progress(train_file_path)
test_data = load_battles_with_progress(test_file_path)

print(f"Training battles: {len(train_data)}")
print(f"Test battles: {len(test_data)}")

# --------------------------------------------------------
# 2. Feature Engineering Corretto
# --------------------------------------------------------
def calculate_type_advantage_advanced(p1_types, p2_types):
    """Calcola vantaggio di tipo più realistico"""
    if not p1_types or not p2_types:
        return 0
    
    # Vantaggi di tipo semplificati con multipliers
    type_chart = {
        'fire': {'grass': 2, 'ice': 2, 'bug': 2, 'fire': 0.5, 'water': 0.5, 'rock': 0.5, 'dragon': 0.5},
        'water': {'fire': 2, 'ground': 2, 'rock': 2, 'water': 0.5, 'grass': 0.5, 'dragon': 0.5},
        'grass': {'water': 2, 'ground': 2, 'rock': 2, 'fire': 0.5, 'grass': 0.5, 'poison': 0.5, 'flying': 0.5, 'bug': 0.5, 'dragon': 0.5, 'steel': 0.5},
        'electric': {'water': 2, 'flying': 2, 'electric': 0.5, 'ground': 0},
        'ice': {'grass': 2, 'ground': 2, 'flying': 2, 'dragon': 2, 'fire': 0.5, 'water': 0.5, 'ice': 0.5, 'steel': 0.5},
        'fighting': {'normal': 2, 'ice': 2, 'rock': 2, 'dark': 2, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'fairy': 0.5},
        # ... aggiungere altri tipi analogamente
    }
    
    advantage = 0
    for p1_type in p1_types:
        for p2_type in p2_types:
            multiplier = type_chart.get(p1_type.lower(), {}).get(p2_type.lower(), 1)
            advantage += multiplier
            # Sottrarre il vantaggio inverso di p2 su p1
            multiplier_rev = type_chart.get(p2_type.lower(), {}).get(p1_type.lower(), 1)
            advantage -= multiplier_rev
    return advantage


def extract_advanced_features(battles):
    """Feature engineering di livello competition - CORRETTO"""
    features_list = []
    
    for battle in tqdm(battles, desc="Extracting competition features"):
        features = {}
        
        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details', {})
        
        # Prendi il lead di p1 dal primo pokemon del team (assunzione comune)
        p1_lead = p1_team[0] if p1_team else {}
        
        # === FEATURE BASE MOLTO ESPANSE ===
        # Statistiche team player 1
        if p1_team:
            # Statistiche base
            stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
            for stat in stats:
                values = [p.get(f'base_{stat}', 0) for p in p1_team]
                features[f'p1_avg_{stat}'] = np.mean(values)
                features[f'p1_max_{stat}'] = np.max(values)
                features[f'p1_min_{stat}'] = np.min(values)
                features[f'p1_std_{stat}'] = np.std(values)
                features[f'p1_total_{stat}'] = np.sum(values)
            
            # Features di squadra
            features['p1_team_size'] = len(p1_team)
            features['p1_team_hp_range'] = np.max([p.get('base_hp', 0) for p in p1_team]) - np.min([p.get('base_hp', 0) for p in p1_team])
            features['p1_team_speed_range'] = np.max([p.get('base_spe', 0) for p in p1_team]) - np.min([p.get('base_spe', 0) for p in p1_team])
            
            # Balance metrics
            features['p1_atk_def_ratio'] = features['p1_avg_atk'] / max(1, features['p1_avg_def'])
            features['p1_physical_special_ratio'] = features['p1_avg_atk'] / max(1, features['p1_avg_spa'])
        
        # === FEATURE LEAD POKEMON ===
        if p1_lead:
            for stat in stats:
                features[f'p1_lead_{stat}'] = p1_lead.get(f'base_{stat}', 0)
        
        if p2_lead:
            for stat in stats:
                features[f'p2_lead_{stat}'] = p2_lead.get(f'base_{stat}', 0)
            
            # === FEATURE DI CONFRONTO AVANZATE ===
            if p1_team and p1_lead:
                # Confronto lead vs lead
                for stat in stats:
                    features[f'lead_{stat}_diff'] = p1_lead.get(f'base_{stat}', 0) - p2_lead.get(f'base_{stat}', 0)
                    features[f'lead_{stat}_ratio'] = p1_lead.get(f'base_{stat}', 0) / max(1, p2_lead.get(f'base_{stat}', 0))
                
                # Confronto team vs lead
                for stat in stats:
                    features[f'team_vs_lead_{stat}_diff'] = features[f'p1_avg_{stat}'] - p2_lead.get(f'base_{stat}', 0)
                    features[f'team_vs_lead_{stat}_ratio'] = features[f'p1_avg_{stat}'] / max(1, p2_lead.get(f'base_{stat}', 0))
        
        # === FEATURE BATTLE TIMELINE AVANZATE ===
        timeline = battle.get('battle_timeline', [])
        features['timeline_length'] = len(timeline)
        
        if timeline:
            # Analisi primi turni (cruciali)
            early_turns = timeline[:6]  # Primi 6 turni
            
            p1_moves = [turn for turn in early_turns if turn.get('p1_move')]
            p2_moves = [turn for turn in early_turns if turn.get('p2_move')]
            p1_switches = [turn for turn in early_turns if turn.get('p1_switch')]
            p2_switches = [turn for turn in early_turns if turn.get('p2_switch')]
            
            features['p1_early_moves'] = len(p1_moves)
            features['p2_early_moves'] = len(p2_moves)
            features['p1_early_switches'] = len(p1_switches)
            features['p2_early_switches'] = len(p2_switches)
            features['early_move_balance'] = len(p1_moves) - len(p2_moves)
            features['early_switch_balance'] = len(p1_switches) - len(p2_switches)
            
            # Momentum iniziale
            features['early_action_ratio'] = len(p1_moves) / max(1, len(p2_moves))
            
            # Analizza se il player 1 ha mosso prima nel primo turno
            if timeline:
                first_turn = timeline[0]
                features['p1_moved_first'] = 1 if first_turn.get('p1_move') else 0
        
        # === FEATURE TIPO E ABILITÀ ===
        if p1_lead and p2_lead:
            p1_lead_types = p1_lead.get('types', [])
            p2_lead_types = p2_lead.get('types', [])
            
            features['type_advantage'] = calculate_type_advantage_advanced(p1_lead_types, p2_lead_types)
            features['type_matchup_score'] = features['type_advantage']
        
        # === FEATURE SINTETICHE ===
        # Overall power score
        if p1_team and p2_lead:
            p1_total_power = sum([p.get('base_hp', 0) + p.get('base_atk', 0) + p.get('base_def', 0) + 
                                p.get('base_spa', 0) + p.get('base_spd', 0) + p.get('base_spe', 0) 
                                for p in p1_team])
            p2_lead_power = sum([p2_lead.get(f'base_{stat}', 0) for stat in ['hp', 'atk', 'def', 'spa', 'spd', 'spe']])
            
            features['total_power_ratio'] = p1_total_power / max(1, p2_lead_power * 6)  # Normalizza
            features['speed_advantage'] = 1 if features.get('p1_avg_spe', 0) > features.get('p2_lead_spe', 0) else 0
        
        # Battle ID e target
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])
        
        features_list.append(features)
    
    return pd.DataFrame(features_list).fillna(0)

print("\n2. Creating competition-grade features...")
train_df = extract_advanced_features(train_data)
test_df = extract_advanced_features(test_data)

# --------------------------------------------------------
# 3. Preparazione Dati
# --------------------------------------------------------
print("\n3. Preparing data...")

# Seleziona features escludendo colonne non numeriche
exclude_cols = ['battle_id', 'player_won']
feature_columns = [col for col in train_df.columns if col not in exclude_cols]

X_train = train_df[feature_columns]
y_train = train_df['player_won']
X_test = test_df[feature_columns]

print(f"Total features generated: {len(feature_columns)}")
print(f"Training shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Feature selection semplice - prendi le prime 40 features per performance
if len(feature_columns) > 40:
    selected_features = feature_columns[:40]
else:
    selected_features = feature_columns

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

print(f"Using {len(selected_features)} features")

# --------------------------------------------------------
# 4. Modelli Ensemble Avanzati
# --------------------------------------------------------
print("\n4. Training advanced ensemble models...")

# Definisci i modelli
models = {
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42
    )
}

# Cross-validation avanzata
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}

print("\nCross-validation results:")
for name, model in models.items():
    scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
    cv_scores[name] = scores
    print(f"{name:20}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Ensemble Voting Classifier
print("\nTraining ensemble model...")
ensemble = VotingClassifier(
    estimators=[
        ('gb', models['GradientBoosting']),
        ('rf', models['RandomForest'])
    ],
    voting='soft'
)

# Cross-validation per l'ensemble
ensemble_scores = cross_val_score(ensemble, X_train_selected, y_train, cv=cv, scoring='accuracy')
print(f"Ensemble            : {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std() * 2:.4f})")

# --------------------------------------------------------
# 5. Training Finale e Predizioni
# --------------------------------------------------------
print("\n5. Training final model and generating predictions...")

# Train final ensemble su tutti i dati di training
final_model = ensemble
final_model.fit(X_train_selected, y_train)

# Predizioni finali
test_predictions = final_model.predict(X_test_selected)

# Valutazione interna (sul training set)
train_predictions = final_model.predict(X_train_selected)
train_accuracy = accuracy_score(y_train, train_predictions)

print(f"\nFinal Training Accuracy: {train_accuracy:.4f}")

# --------------------------------------------------------
# 6. Creazione Submission File
# --------------------------------------------------------
print("\n6. Creating submission file...")

# Crea il file di submission
submission_df = pd.DataFrame({
    'battle_id': test_df['battle_id'],
    'player_won': test_predictions
})

# Salva il file
submission_path = 'competition_submission.csv'
submission_df.to_csv(submission_path, index=False)

print(f"✅ Submission file saved: {submission_path}")
print(f"✅ File location: {os.path.abspath(submission_path)}")

# --------------------------------------------------------
# 7. Analisi Dettagliata
# --------------------------------------------------------
print("\n" + "=" * 70)
print("DETAILED ANALYSIS")
print("=" * 70)

print(f"Total battles processed: {len(train_data) + len(test_data)}")
print(f"Final features used: {len(selected_features)}")
print(f"Ensemble CV Score: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std() * 2:.4f})")
print(f"Training Accuracy: {train_accuracy:.4f}")

# Distribuzione predizioni
win_rate = np.mean(test_predictions)
print(f"📊 Predicted win rate: {win_rate:.2%}")

# Anteprima features
print(f"\n📋 First 10 features: {selected_features[:10]}")

print("\n" + "=" * 70)
print("COMPETITION SUBMISSION READY!")
print("=" * 70)
