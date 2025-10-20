# --- 1. Import delle Librerie Necessarie ---
import json
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- 2. Definizione delle Costanti e dei Percorsi ---
# Modifica questo percorso per puntare alla cartella contenente i tuoi file
DATA_PATH = r"C:\Users\giann\Desktop\universita\magistrale\FUNDATIONS OF DATA SCIENCE\progetto pokemon"
TRAIN_FILE_PATH = os.path.join(DATA_PATH, 'train.jsonl')
TEST_FILE_PATH = os.path.join(DATA_PATH, 'test.jsonl')
SUBMISSION_FILE_PATH = os.path.join(DATA_PATH, 'submission.csv')

# Mappa delle efficacie dei tipi di Pokémon
TYPE_CHART = {
    'normal': {'rock': 0.5, 'ghost': 0, 'steel': 0.5}, 'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 2, 'bug': 2, 'rock': 0.5, 'dragon': 0.5, 'steel': 2}, 'water': {'fire': 2, 'water': 0.5, 'grass': 0.5, 'ground': 2, 'rock': 2, 'dragon': 0.5}, 'electric': {'water': 2, 'electric': 0.5, 'grass': 0.5, 'ground': 0, 'flying': 2, 'dragon': 0.5}, 'grass': {'fire': 0.5, 'water': 2, 'grass': 0.5, 'poison': 0.5, 'ground': 2, 'flying': 0.5, 'bug': 0.5, 'rock': 2, 'dragon': 0.5, 'steel': 0.5}, 'ice': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 0.5, 'ground': 2, 'flying': 2, 'dragon': 2, 'steel': 0.5}, 'fighting': {'normal': 2, 'ice': 2, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2, 'ghost': 0, 'dark': 2, 'steel': 2, 'fairy': 0.5}, 'poison': {'grass': 2, 'poison': 0.5, 'ground': 0.5, 'rock': 0.5, 'ghost': 0.5, 'steel': 0, 'fairy': 2}, 'ground': {'fire': 2, 'electric': 2, 'grass': 0.5, 'poison': 2, 'flying': 0, 'bug': 0.5, 'rock': 2, 'steel': 2}, 'flying': {'electric': 0.5, 'grass': 2, 'fighting': 2, 'bug': 2, 'rock': 0.5, 'steel': 0.5}, 'psychic': {'fighting': 2, 'poison': 2, 'psychic': 0.5, 'dark': 0, 'steel': 0.5}, 'bug': {'fire': 0.5, 'grass': 2, 'fighting': 0.5, 'poison': 0.5, 'flying': 0.5, 'psychic': 2, 'ghost': 0.5, 'dark': 2, 'steel': 0.5, 'fairy': 0.5}, 'rock': {'fire': 2, 'ice': 2, 'fighting': 0.5, 'ground': 0.5, 'flying': 2, 'bug': 2, 'steel': 0.5}, 'ghost': {'normal': 0, 'psychic': 2, 'ghost': 2, 'dark': 0.5}, 'dragon': {'dragon': 2, 'steel': 0.5, 'fairy': 0}, 'dark': {'fighting': 0.5, 'psychic': 2, 'ghost': 2, 'dark': 0.5, 'fairy': 0.5}, 'steel': {'fire': 0.5, 'water': 0.5, 'electric': 0.5, 'ice': 2, 'rock': 2, 'steel': 0.5, 'fairy': 2}, 'fairy': {'fire': 0.5, 'fighting': 2, 'poison': 0.5, 'dragon': 2, 'dark': 2, 'steel': 0.5}
}


# --- 3. Funzioni per l'Ingegneria delle Feature (Feature Engineering) ---

def get_type_effectiveness(move_type: str, target_types: list[str]) -> float:
    """Calcola il moltiplicatore di efficacia di un tipo di mossa contro i tipi del bersaglio."""
    if not move_type or not target_types:
        return 1.0
    
    move_type = move_type.lower()
    target_types = [t.lower() for t in target_types if t and t.lower() != 'notype']
    
    if move_type not in TYPE_CHART or not target_types:
        return 1.0

    effectiveness = 1.0
    for target_type in target_types:
        effectiveness *= TYPE_CHART.get(move_type, {}).get(target_type, 1.0)
    return effectiveness

def create_battle_features(data: list[dict]) -> pd.DataFrame:
    """
    Estrae un set completo di feature da una lista di battaglie.
    Include analisi del team, matchup iniziale e aggregazione di eventi su tutta la timeline.
    """
    feature_list = []
    for battle in tqdm(data, desc="Extracting battle features"):
        features = {}

        # Parte 1: Analisi del team e del matchup iniziale
        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details')

        if p1_team:
            features['p1_mean_hp'] = np.mean([p.get('base_hp', 0) for p in p1_team])
            features['p1_mean_atk'] = np.mean([p.get('base_atk', 0) for p in p1_team])
            features['p1_mean_def'] = np.mean([p.get('base_def', 0) for p in p1_team])
            features['p1_mean_spa'] = np.mean([p.get('base_spa', 0) for p in p1_team])
            features['p1_mean_spd'] = np.mean([p.get('base_spd', 0) for p in p1_team])
            features['p1_mean_spe'] = np.mean([p.get('base_spe', 0) for p in p1_team])
            features['p1_max_spe'] = np.max([p.get('base_spe', 0) for p in p1_team])

        if p2_lead:
            features['p2_lead_spe'] = p2_lead.get('base_spe', 0)
        
        if p1_team and p2_lead:
            features['speed_advantage'] = features.get('p1_max_spe', 0) - features.get('p2_lead_spe', 0)
            
            max_eff = max(
                (get_type_effectiveness(p1_type, p2_lead.get('types', []))
                 for p1_pkmn in p1_team for p1_type in p1_pkmn.get('types', [])),
                default=1.0
            )
            features['p1_max_effectiveness_vs_lead'] = max_eff

        # Parte 2: Analisi aggregata dell'intera battaglia
        timeline = battle.get('battle_timeline')
        
        # Inizializza le feature aggregate a 0
        aggr_features = {
            'battle_duration_turns': 0, 'p1_total_damage_dealt': 0.0, 'p2_total_damage_dealt': 0.0,
            'p1_ko_count': 0, 'p2_ko_count': 0, 'p1_switch_count': 0, 'p2_switch_count': 0,
            'p1_total_boosts': 0, 'p2_total_boosts': 0
        }

        if timeline:
            aggr_features['battle_duration_turns'] = len(timeline)
            last_known_hp = {}
            fainted_by_p1, fainted_by_p2 = set(), set()

            for turn in timeline:
                p1_state, p2_state = turn.get('p1_pokemon_state'), turn.get('p2_pokemon_state')
                
                if p2_state:
                    name, hp = p2_state['name'], p2_state.get('hp_pct', 1.0)
                    dmg = last_known_hp.get(name, 1.0) - hp
                    if dmg > 0: aggr_features['p1_total_damage_dealt'] += dmg
                    last_known_hp[name] = hp
                    if hp == 0 and name not in fainted_by_p1:
                        aggr_features['p1_ko_count'] += 1
                        fainted_by_p1.add(name)
                
                if p1_state:
                    name, hp = p1_state['name'], p1_state.get('hp_pct', 1.0)
                    dmg = last_known_hp.get(name, 1.0) - hp
                    if dmg > 0: aggr_features['p2_total_damage_dealt'] += dmg
                    last_known_hp[name] = hp
                    if hp == 0 and name not in fainted_by_p2:
                        aggr_features['p2_ko_count'] += 1
                        fainted_by_p2.add(name)

                if not turn.get('p1_move_details'): aggr_features['p1_switch_count'] += 1
                if not turn.get('p2_move_details'): aggr_features['p2_switch_count'] += 1
                
                if p1_state and p1_state.get('boosts'):
                    aggr_features['p1_total_boosts'] += sum(v for v in p1_state['boosts'].values() if v > 0)
                if p2_state and p2_state.get('boosts'):
                    aggr_features['p2_total_boosts'] += sum(v for v in p2_state['boosts'].values() if v > 0)
        
        features.update(aggr_features)

        # Parte 3: Creazione di feature comparative
        features['ko_advantage'] = features['p1_ko_count'] - features['p2_ko_count']
        features['net_damage_advantage'] = features['p1_total_damage_dealt'] - features['p2_total_damage_dealt']
        features['net_boost_advantage'] = features['p1_total_boosts'] - features['p2_total_boosts']

        # Parte 4: Aggiunta ID e Target
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])
            
        feature_list.append(features)
        
    return pd.DataFrame(feature_list).fillna(0)


# --- 4. Esecuzione del Flusso di Lavoro di Machine Learning ---

# Caricamento dei dati grezzi
print(f"Loading training data from '{TRAIN_FILE_PATH}'...")
try:
    with open(TRAIN_FILE_PATH, 'r') as f:
        train_data_raw = [json.loads(line) for line in f]
    print(f"Successfully loaded {len(train_data_raw)} training battles.")
except FileNotFoundError:
    print(f"ERROR: Training file not found at '{TRAIN_FILE_PATH}'.")
    exit()

# Creazione del DataFrame di training con le nuove feature
train_df = create_battle_features(train_data_raw)

# Preparazione del set di training per il modello
features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
X = train_df[features]
y = train_df['player_won']

# Divisione in set di training e di validazione per una stima accurata delle performance
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData split: {len(X_train)} training samples, {len(X_val)} validation samples.")

# Addestramento del modello LightGBM
print("\nTraining LightGBM model...")
model = lgb.LGBMClassifier(random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=40)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='accuracy',
    callbacks=[lgb.early_stopping(10, verbose=False)] # Stoppa se l'accuracy non migliora per 10 iterazioni
)
print("Model training complete.")

# Valutazione del modello sul set di validazione
val_accuracy = model.score(X_val, y_val)
print(f"\n--- Model Performance ---")
print(f"Accuracy on Validation Set: {val_accuracy:.4f} ({val_accuracy:.2%})")

# --- 5. Generazione delle Predizioni e Creazione della Submission ---

# Caricamento e processamento dei dati di test
print(f"\nLoading and processing test data from '{TEST_FILE_PATH}'...")
try:
    with open(TEST_FILE_PATH, 'r') as f:
        test_data_raw = [json.loads(line) for line in f]
    test_df = create_battle_features(test_data_raw)
except FileNotFoundError:
    print(f"ERROR: Test file not found at '{TEST_FILE_PATH}'.")
    exit()

# **Importante**: Salva gli ID prima di selezionare le feature per la predizione
test_ids = test_df['battle_id']
X_test = test_df[features] # Assicura che le colonne corrispondano a quelle di training

# Generazione delle predizioni
print("Generating predictions on the test set...")
test_predictions = model.predict(X_test)
test_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilità di vittoria

# Creazione del DataFrame per la submission
submission_df = pd.DataFrame({
    'battle_id': test_ids,
    'player_won': test_predictions # Usa le probabilità per un ranking più fine
})

# Salvataggio del file
submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)
print(f"\nSubmission file '{os.path.basename(SUBMISSION_FILE_PATH)}' created successfully!")
print("--- Submission Head ---")
print(submission_df.head())
