# --- 1. Import delle Librerie Necessarie ---
import json
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- 2. Definizione delle Costanti e dei Percorsi ---
DATA_PATH = r"C:\Users\franc\Desktop\data science\PRIMO ANNO\python\pokemon-fds"
TRAIN_FILE_PATH = os.path.join(DATA_PATH, 'train.jsonl')
TEST_FILE_PATH = os.path.join(DATA_PATH, 'test.jsonl')
SUBMISSION_FILE_PATH = os.path.join(DATA_PATH, 'submission.csv')

# Mappa delle efficacie dei tipi di Pokémon
TYPE_CHART = {
    'normal': {'rock': 0.5, 'ghost': 0, 'steel': 0.5},
    'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 2, 'bug': 2, 'rock': 0.5, 'dragon': 0.5, 'steel': 2}, 
    'water': {'fire': 2, 'water': 0.5, 'grass': 0.5, 'ground': 2, 'rock': 2, 'dragon': 0.5}, 
    'electric': {'water': 2, 'electric': 0.5, 'grass': 0.5, 'ground': 0, 'flying': 2, 'dragon': 0.5}, 
    'grass': {'fire': 0.5, 'water': 2, 'grass': 0.5, 'poison': 0.5, 'ground': 2, 'flying': 0.5, 'bug': 0.5, 'rock': 2, 'dragon': 0.5, 'steel': 0.5}, 
    'ice': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 0.5, 'ground': 2, 'flying': 2, 'dragon': 2, 'steel': 0.5}, 
    'fighting': {'normal': 2, 'ice': 2, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2, 'ghost': 0, 'dark': 2, 'steel': 2, 'fairy': 0.5}, 
    'poison': {'grass': 2, 'poison': 0.5, 'ground': 0.5, 'rock': 0.5, 'ghost': 0.5, 'steel': 0, 'fairy': 2}, 
    'ground': {'fire': 2, 'electric': 2, 'grass': 0.5, 'poison': 2, 'flying': 0, 'bug': 0.5, 'rock': 2, 'steel': 2}, 
    'flying': {'electric': 0.5, 'grass': 2, 'fighting': 2, 'bug': 2, 'rock': 0.5, 'steel': 0.5}, 
    'psychic': {'fighting': 2, 'poison': 2, 'psychic': 0.5, 'dark': 0, 'steel': 0.5}, 
    'bug': {'fire': 0.5, 'grass': 2, 'fighting': 0.5, 'poison': 0.5, 'flying': 0.5, 'psychic': 2, 'ghost': 0.5, 'dark': 2, 'steel': 0.5, 'fairy': 0.5}, 
    'rock': {'fire': 2, 'ice': 2, 'fighting': 0.5, 'ground': 0.5, 'flying': 2, 'bug': 2, 'steel': 0.5}, 
    'ghost': {'normal': 0, 'psychic': 2, 'ghost': 2, 'dark': 0.5}, 
    'dragon': {'dragon': 2, 'steel': 0.5, 'fairy': 0}, 
    'dark': {'fighting': 0.5, 'psychic': 2, 'ghost': 2, 'dark': 0.5, 'fairy': 0.5}, 
    'steel': {'fire': 0.5, 'water': 0.5, 'electric': 0.5, 'ice': 2, 'rock': 2, 'steel': 0.5, 'fairy': 2}, 
    'fairy': {'fire': 0.5, 'fighting': 2, 'poison': 0.5, 'dragon': 2, 'dark': 2, 'steel': 0.5}
}

# --- 3. Funzioni per l'Ingegneria delle Feature ---

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
    """Estrae feature avanzate dalle battaglie Pokemon."""
    feature_list = []
    
    for battle in tqdm(data, desc="Extracting battle features"):
        features = {}

        # === PARTE 1: Analisi Team Completa ===
        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details')

        if p1_team:
            # Statistiche base del team
            features['p1_mean_hp'] = np.mean([p.get('base_hp', 0) for p in p1_team])
            features['p1_mean_atk'] = np.mean([p.get('base_atk', 0) for p in p1_team])
            features['p1_mean_def'] = np.mean([p.get('base_def', 0) for p in p1_team])
            features['p1_mean_spa'] = np.mean([p.get('base_spa', 0) for p in p1_team])
            features['p1_mean_spd'] = np.mean([p.get('base_spd', 0) for p in p1_team])
            features['p1_mean_spe'] = np.mean([p.get('base_spe', 0) for p in p1_team])
            
            # NUOVO: Statistiche minime e massime
            features['p1_max_hp'] = np.max([p.get('base_hp', 0) for p in p1_team])
            features['p1_min_hp'] = np.min([p.get('base_hp', 0) for p in p1_team])
            features['p1_max_spe'] = np.max([p.get('base_spe', 0) for p in p1_team])
            features['p1_min_spe'] = np.min([p.get('base_spe', 0) for p in p1_team])
            
            # NUOVO: Deviazione standard (variabilità del team)
            features['p1_std_spe'] = np.std([p.get('base_spe', 0) for p in p1_team])
            features['p1_std_hp'] = np.std([p.get('base_hp', 0) for p in p1_team])
            
            # NUOVO: Statistiche offensive/difensive combinate
            features['p1_mean_offensive'] = np.mean([
                max(p.get('base_atk', 0), p.get('base_spa', 0)) for p in p1_team
            ])
            features['p1_mean_defensive'] = np.mean([
                (p.get('base_def', 0) + p.get('base_spd', 0)) / 2 for p in p1_team
            ])
            features['p1_max_offensive'] = np.max([
                max(p.get('base_atk', 0), p.get('base_spa', 0)) for p in p1_team
            ])
            
            # NUOVO: BST (Base Stat Total)
            features['p1_mean_bst'] = np.mean([
                sum([p.get('base_hp', 0), p.get('base_atk', 0), p.get('base_def', 0),
                     p.get('base_spa', 0), p.get('base_spd', 0), p.get('base_spe', 0)])
                for p in p1_team
            ])
            features['p1_max_bst'] = np.max([
                sum([p.get('base_hp', 0), p.get('base_atk', 0), p.get('base_def', 0),
                     p.get('base_spa', 0), p.get('base_spd', 0), p.get('base_spe', 0)])
                for p in p1_team
            ])

        if p2_lead:
            features['p2_lead_spe'] = p2_lead.get('base_spe', 0)
            features['p2_lead_hp'] = p2_lead.get('base_hp', 0)
            features['p2_lead_offensive'] = max(p2_lead.get('base_atk', 0), p2_lead.get('base_spa', 0))
            features['p2_lead_defensive'] = (p2_lead.get('base_def', 0) + p2_lead.get('base_spd', 0)) / 2
            features['p2_lead_bst'] = sum([
                p2_lead.get('base_hp', 0), p2_lead.get('base_atk', 0), p2_lead.get('base_def', 0),
                p2_lead.get('base_spa', 0), p2_lead.get('base_spd', 0), p2_lead.get('base_spe', 0)
            ])
        
        # Matchup features
        if p1_team and p2_lead:
            features['speed_advantage'] = features.get('p1_max_spe', 0) - features.get('p2_lead_spe', 0)
            features['offensive_advantage'] = features.get('p1_max_offensive', 0) - features.get('p2_lead_offensive', 0)
            features['bst_advantage'] = features.get('p1_max_bst', 0) - features.get('p2_lead_bst', 0)
            
            # Efficacia di tipo
            max_eff = max(
                (get_type_effectiveness(p1_type, p2_lead.get('types', []))
                 for p1_pkmn in p1_team for p1_type in p1_pkmn.get('types', [])),
                default=1.0
            )
            features['p1_max_effectiveness_vs_lead'] = max_eff
            
            # NUOVO: Conta quanti Pokemon hanno vantaggio di tipo
            features['p1_type_advantage_count'] = sum(
                1 for p1_pkmn in p1_team
                for p1_type in p1_pkmn.get('types', [])
                if get_type_effectiveness(p1_type, p2_lead.get('types', [])) > 1.0
            )

        # === PARTE 2: Analisi Timeline Dettagliata ===
        timeline = battle.get('battle_timeline', [])
        
        aggr_features = {
            'battle_duration_turns': 0, 'p1_total_damage_dealt': 0.0, 'p2_total_damage_dealt': 0.0,
            'p1_ko_count': 0, 'p2_ko_count': 0, 'p1_switch_count': 0, 'p2_switch_count': 0,
            'p1_total_boosts': 0, 'p2_total_boosts': 0, 'p1_total_negative_boosts': 0, 'p2_total_negative_boosts': 0,
            'p1_status_moves': 0, 'p2_status_moves': 0, 'p1_attacking_moves': 0, 'p2_attacking_moves': 0,
            'p1_paralysis_count': 0, 'p2_paralysis_count': 0, 'p1_burn_count': 0, 'p2_burn_count': 0,
            'p1_poison_count': 0, 'p2_poison_count': 0, 'p1_sleep_count': 0, 'p2_sleep_count': 0,
            'p1_early_damage': 0.0, 'p2_early_damage': 0.0, 'p1_late_damage': 0.0, 'p2_late_damage': 0.0,
            'p1_super_effective_count': 0, 'p2_super_effective_count': 0,
            'p1_unique_pokemon_used': 0, 'p2_unique_pokemon_used': 0
        }

        if timeline:
            aggr_features['battle_duration_turns'] = len(timeline)
            last_known_hp = {}
            fainted_by_p1, fainted_by_p2 = set(), set()
            p1_pokemon_seen, p2_pokemon_seen = set(), set()
            mid_turn = len(timeline) // 2

            for idx, turn in enumerate(timeline):
                p1_state = turn.get('p1_pokemon_state')
                p2_state = turn.get('p2_pokemon_state')
                p1_move = turn.get('p1_move_details')
                p2_move = turn.get('p2_move_details')
                
                # Traccia Pokemon unici usati
                if p1_state:
                    p1_pokemon_seen.add(p1_state['name'])
                if p2_state:
                    p2_pokemon_seen.add(p2_state['name'])
                
                # Analisi danno P2 (inflitto da P1)
                if p2_state:
                    name, hp = p2_state['name'], p2_state.get('hp_pct', 1.0)
                    dmg = last_known_hp.get(name, 1.0) - hp
                    if dmg > 0:
                        aggr_features['p1_total_damage_dealt'] += dmg
                        if idx < mid_turn:
                            aggr_features['p1_early_damage'] += dmg
                        else:
                            aggr_features['p1_late_damage'] += dmg
                    last_known_hp[name] = hp
                    
                    if hp == 0 and name not in fainted_by_p1:
                        aggr_features['p1_ko_count'] += 1
                        fainted_by_p1.add(name)
                    
                    # Status analysis
                    status = p2_state.get('status', 'nostatus')
                    if status == 'par': aggr_features['p2_paralysis_count'] += 1
                    elif status == 'brn': aggr_features['p2_burn_count'] += 1
                    elif status in ['psn', 'tox']: aggr_features['p2_poison_count'] += 1
                    elif status == 'slp': aggr_features['p2_sleep_count'] += 1
                
                # Analisi danno P1 (inflitto da P2)
                if p1_state:
                    name, hp = p1_state['name'], p1_state.get('hp_pct', 1.0)
                    dmg = last_known_hp.get(name, 1.0) - hp
                    if dmg > 0:
                        aggr_features['p2_total_damage_dealt'] += dmg
                        if idx < mid_turn:
                            aggr_features['p2_early_damage'] += dmg
                        else:
                            aggr_features['p2_late_damage'] += dmg
                    last_known_hp[name] = hp
                    
                    if hp == 0 and name not in fainted_by_p2:
                        aggr_features['p2_ko_count'] += 1
                        fainted_by_p2.add(name)
                    
                    status = p1_state.get('status', 'nostatus')
                    if status == 'par': aggr_features['p1_paralysis_count'] += 1
                    elif status == 'brn': aggr_features['p1_burn_count'] += 1
                    elif status in ['psn', 'tox']: aggr_features['p1_poison_count'] += 1
                    elif status == 'slp': aggr_features['p1_sleep_count'] += 1

                # Switch count
                if not p1_move: aggr_features['p1_switch_count'] += 1
                if not p2_move: aggr_features['p2_switch_count'] += 1
                
                # Move analysis
                if p1_move:
                    if p1_move.get('category') == 'STATUS':
                        aggr_features['p1_status_moves'] += 1
                    else:
                        aggr_features['p1_attacking_moves'] += 1
                        
                    # Super effective check
                    if p2_state and p1_move.get('type'):
                        eff = get_type_effectiveness(
                            p1_move['type'].lower(),
                            [t for t in p2_state.get('types', []) if t]
                        )
                        if eff > 1.0:
                            aggr_features['p1_super_effective_count'] += 1
                
                if p2_move:
                    if p2_move.get('category') == 'STATUS':
                        aggr_features['p2_status_moves'] += 1
                    else:
                        aggr_features['p2_attacking_moves'] += 1
                        
                    if p1_state and p2_move.get('type'):
                        eff = get_type_effectiveness(
                            p2_move['type'].lower(),
                            [t for t in p1_state.get('types', []) if t]
                        )
                        if eff > 1.0:
                            aggr_features['p2_super_effective_count'] += 1
                
                # Boost analysis
                if p1_state and p1_state.get('boosts'):
                    for v in p1_state['boosts'].values():
                        if v > 0:
                            aggr_features['p1_total_boosts'] += v
                        elif v < 0:
                            aggr_features['p1_total_negative_boosts'] += abs(v)
                            
                if p2_state and p2_state.get('boosts'):
                    for v in p2_state['boosts'].values():
                        if v > 0:
                            aggr_features['p2_total_boosts'] += v
                        elif v < 0:
                            aggr_features['p2_total_negative_boosts'] += abs(v)
            
            aggr_features['p1_unique_pokemon_used'] = len(p1_pokemon_seen)
            aggr_features['p2_unique_pokemon_used'] = len(p2_pokemon_seen)
        
        features.update(aggr_features)

        # === PARTE 3: Feature Comparative e Ratios ===
        features['ko_advantage'] = features['p1_ko_count'] - features['p2_ko_count']
        features['net_damage_advantage'] = features['p1_total_damage_dealt'] - features['p2_total_damage_dealt']
        features['net_boost_advantage'] = features['p1_total_boosts'] - features['p2_total_boosts']
        features['switch_difference'] = features['p1_switch_count'] - features['p2_switch_count']
        features['status_move_difference'] = features['p1_status_moves'] - features['p2_status_moves']
        features['super_effective_difference'] = features['p1_super_effective_count'] - features['p2_super_effective_count']
        
        # Ratios (evita divisione per zero)
        if features['p2_total_damage_dealt'] > 0:
            features['damage_ratio'] = features['p1_total_damage_dealt'] / features['p2_total_damage_dealt']
        else:
            features['damage_ratio'] = features['p1_total_damage_dealt']
        
        if features['battle_duration_turns'] > 0:
            features['p1_ko_rate'] = features['p1_ko_count'] / features['battle_duration_turns']
            features['p2_ko_rate'] = features['p2_ko_count'] / features['battle_duration_turns']
            features['p1_damage_per_turn'] = features['p1_total_damage_dealt'] / features['battle_duration_turns']
            features['p2_damage_per_turn'] = features['p2_total_damage_dealt'] / features['battle_duration_turns']
        
        # Momentum features
        if features['p1_early_damage'] > 0:
            features['p1_momentum'] = features['p1_late_damage'] / features['p1_early_damage']
        else:
            features['p1_momentum'] = features['p1_late_damage']
            
        if features['p2_early_damage'] > 0:
            features['p2_momentum'] = features['p2_late_damage'] / features['p2_early_damage']
        else:
            features['p2_momentum'] = features['p2_late_damage']

        # === PARTE 4: ID e Target ===
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])
            
        feature_list.append(features)
        
    return pd.DataFrame(feature_list).fillna(0)


# --- 4. Esecuzione del Flusso di Lavoro ---

print(f"Loading training data from '{TRAIN_FILE_PATH}'...")
try:
    with open(TRAIN_FILE_PATH, 'r') as f:
        train_data_raw = [json.loads(line) for line in f]
    print(f"Successfully loaded {len(train_data_raw)} training battles.")
except FileNotFoundError:
    print(f"ERROR: Training file not found at '{TRAIN_FILE_PATH}'.")
    exit()

train_df = create_battle_features(train_data_raw)

features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
X = train_df[features]
y = train_df['player_won']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData split: {len(X_train)} training samples, {len(X_val)} validation samples.")
print(f"Total features: {len(features)}")

# MIGLIORAMENTO: Parametri ottimizzati per LightGBM
print("\nTraining LightGBM model with optimized parameters...")
model = lgb.LGBMClassifier(
    random_state=42,
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=50,
    max_depth=8,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    importance_type='gain'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='accuracy',
    callbacks=[lgb.early_stopping(50, verbose=False)]
)
print("Model training complete.")

val_accuracy = model.score(X_val, y_val)
print(f"\n--- Model Performance ---")
print(f"Accuracy on Validation Set: {val_accuracy:.4f} ({val_accuracy:.2%})")

# Feature importance
print("\n--- Top 15 Most Important Features ---")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(15))

# --- 5. Generazione Predizioni ---

print(f"\nLoading and processing test data from '{TEST_FILE_PATH}'...")
try:
    with open(TEST_FILE_PATH, 'r') as f:
        test_data_raw = [json.loads(line) for line in f]
    test_df = create_battle_features(test_data_raw)
except FileNotFoundError:
    print(f"ERROR: Test file not found at '{TEST_FILE_PATH}'.")
    exit()

test_ids = test_df['battle_id']
X_test = test_df[features]

print("Generating predictions on the test set...")
test_predictions = model.predict(X_test)

submission_df = pd.DataFrame({
    'battle_id': test_ids,
    'player_won': test_predictions
})

submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)
print(f"\nSubmission file '{os.path.basename(SUBMISSION_FILE_PATH)}' created successfully!")
print("--- Submission Head ---")
print(submission_df.head())
