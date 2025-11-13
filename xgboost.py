import json 
import os  
from tqdm import tqdm  
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
from pprint import pprint
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings("ignore")

COMPETITION_NAME = 'fds-pokemon-battles-prediction-2025'
DATA_PATH = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(DATA_PATH, 'train.jsonl')
test_file_path = os.path.join(DATA_PATH, 'test.jsonl')
train_data = [] 

print(f"Loading data from '{train_file_path}'...")
try:
    with open(train_file_path, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))
    print(f"Successfully loaded {len(train_data)} battles.")
except FileNotFoundError:
    print(f"ERROR: Could not find the training file at '{train_file_path}'.")
    print("Please make sure you have added the competition data to this notebook.")
from typing import List, Dict

def clean_battles(data: List[Dict]) -> List[Dict]:
    cleaned_data = []
    fixed_count = 0
    for battle in data:
        p1_team = battle.get('p1_team_details', [])
        if not isinstance(p1_team, list):
            p1_team = []
        for pkmn in p1_team:
            if pkmn.get('level') != 100:
                pkmn['level'] = 100
                fixed_count += 1
        battle['p1_team_details'] = p1_team
        p2_lead = battle.get('p2_lead_details', {})
        default_level = 100
        if 'level' not in p2_lead or (isinstance(p2_lead.get('level'), (int, float)) and p2_lead['level'] <= 0):
            p2_lead['level'] = default_level
            fixed_count += 1
        battle['p2_lead_details'] = p2_lead
        cleaned_data.append(battle)
    print(f"✅ Pulizia completata: {len(data)} battaglie processate, {fixed_count} correzioni effettuate.")
    return cleaned_data
# --------------------------------------------------------
def clean_battles(data: List[Dict]) -> List[Dict]:
    cleaned_data = []
    fixed_count = 0
    for battle in data:
        p1_team = battle.get('p1_team_details', [])
        if not isinstance(p1_team, list):
            p1_team = []
        for pkmn in p1_team:
            if pkmn.get('level') != 100:
                pkmn['level'] = 100
                fixed_count += 1
        battle['p1_team_details'] = p1_team
        p2_lead = battle.get('p2_lead_details', {})
        default_level = 100
        if 'level' not in p2_lead or (isinstance(p2_lead.get('level'), (int, float)) and p2_lead['level'] <= 0):
            p2_lead['level'] = default_level
            fixed_count += 1
        battle['p2_lead_details'] = p2_lead
        cleaned_data.append(battle)
    print(f"*** Cleaning complete: {len(data)} processed battles, {fixed_count} corrected battles ***")
    return cleaned_data

def create_battle_features(data: list[dict]) -> pd.DataFrame:
    feature_list = []
    for battle in tqdm(data, desc="Feature extraction"): 
        features = {}
        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details', {})
        timeline = battle.get('battle_timeline', [])
        
        #aggr = dictionary holding total counts and measures like total turns, damage dealt/received
        aggr = {'turns':0, 'p1_dmg':0, 'p2_dmg':0, 'p1_ko':0, 'p2_ko':0, 'p2_switch':0, 'p1_heal':0, 'p2_heal':0}
        #last_hp = dict that stores the hp percentage of each pokemon at the end of the previous turn
        last_hp = {}
        #sets that storing the names of the Pokemon belonging to P1 or P2 that have already been KO
        ko_p1, ko_p2 = set(), set()
        #sets storing the neames of Pok from each team that appeared on the field at any point
        p1_pokemon_seen, p2_pokemon_seen = set(), set()
        #lists of the total damage delt by each player during each individual turn
        p1_damage_by_turn, p2_damage_by_turn = [], []
        #lists of the total HP percentage of the active pok for each player at the end of each individual turn
        p1_hp_by_turn, p2_hp_by_turn = [], []     
        #Count of moves that dealt damage greater than 25%
        p1_effective_moves, p2_effective_moves = 0, 0
        #Count of moves used by each player that are classified as PHISYCAL OR SPECIAL
        p1_total_moves, p2_total_moves = 0, 0
        #first_blood = tracks the player who first inflicted significant damage > 15%
        first_blood = None 
        #maximum cumultative HP advantage achieved by each player at any point in the battle
        p1_max_lead, p2_max_lead = 0, 0        
        #flag set to True if a player, who was behind in HP lead, eventually won the KO
        comeback_detected = False
        #ordered list storing which player scored a KO in chronological order
        ko_sequence = []
        #total number of times each player switched out their active pokemon
        p1_switch_count, p2_switch_count = 0, 0
        #total turns where a player's active pok was affected by a major status condition (burn, poison)
        p1_status_turns, p2_status_turns = 0, 0
        #turn number and player who achieved the very first KO of the match
        first_ko_turn, first_ko_by = None, None
        
        for turn_idx, turn in enumerate(timeline):
            turn_num = turn_idx + 1
            #cumultative damage dealt by both players during this specific turn
            p1_dmg_this_turn, p2_dmg_this_turn = 0, 0
            #total HP percentage of the active Pokemon for both players
            p1_hp_this_turn, p2_hp_this_turn = 0, 0 
            for side in ['p1', 'p2']:
                state = turn.get(f'{side}_pokemon_state', {})
                move = turn.get(f'{side}_move_details')
                opp = 'p2' if side == 'p1' else 'p1'
                if state:
                    poke_name = state.get('name', '')
                    cur = state.get('hp_pct', 1.0)
                    prev = last_hp.get(f'{side}_{poke_name}', 1.0)
                    #calculate damage and healing based on the difference between the precious and current HP values
                    dmg = max(prev - cur, 0)
                    heal = max(cur - prev, 0)
                    if side == 'p1':
                        p1_pokemon_seen.add(poke_name)
                        #Adds the current HP perentage(cur) to the temporary turn accumulator
                        p1_hp_this_turn += cur
                    else:
                        p2_pokemon_seen.add(poke_name)
                        p2_hp_this_turn += cur
                    if dmg > 0:
                        #accomulate the total raw damage dealt by each player throughout the entire battle
                        aggr[f'{opp}_dmg'] += dmg
                        #if damage was inflicted by p1
                        if opp == 'p1':
                            p1_dmg_this_turn += dmg
                            if first_blood is None and dmg > 0.15:#only a meaningful attack is counted, ignoring small damage
                                first_blood = 'p1'
                        else:
                            p2_dmg_this_turn += dmg
                            if first_blood is None and dmg > 0.15:
                                first_blood = 'p2'
                    if heal > 0:
                        aggr[f'{side}_heal'] += heal
                    #check if the player used offensive move
                    if move and move.get('category') in ['PHYSICAL', 'SPECIAL']:
                        if side == 'p1':
                            p1_total_moves += 1
                            #if the move caused significant damage it counts as an effective move
                            if dmg > 0.25:
                                p1_effective_moves += 1
                        else:
                            p2_total_moves += 1
                            if dmg > 0.25:
                                p2_effective_moves += 1
                    #if the active Pok has a major status condition, the count of tourns spent undeer status for that player is incremented (fnt = svenimento/KO, quando HP pokemon scendono a 0)
                    status = state.get('status', 'nostatus')
                    if status not in ['nostatus', 'fnt']:
                        if side == 'p1':
                            p1_status_turns += 1
                        else:
                            p2_status_turns += 1
                    #update the dictionary with the Pok's current HP
                    last_hp[f'{side}_{poke_name}'] = cur
                    #set the first ko turn and recors which player scored it
                    if cur == 0:
                        if first_ko_turn is None:
                            first_ko_turn = turn_num
                            first_ko_by = opp
                        #check if the KO has already been counted; if it's a new one, it will be incremented, add pok's name to the KO set and append the opponent's id to the list ko_sequence
                        if side == 'p1' and poke_name not in ko_p2:
                            aggr['p2_ko'] += 1
                            ko_p2.add(poke_name)
                            ko_sequence.append('p2')
                        elif side == 'p2' and poke_name not in ko_p1:
                            aggr['p1_ko'] += 1
                            ko_p1.add(poke_name)
                            ko_sequence.append('p1')
                #if the player did not use a move, there's a switch
                if not move:
                    if side == 'p1':
                        p1_switch_count += 1
                    else:
                        p2_switch_count += 1
                        aggr['p2_switch'] += 1
            #append the temporary pre-turn totals to their respective lists
            p1_damage_by_turn.append(p1_dmg_this_turn)
            p2_damage_by_turn.append(p2_dmg_this_turn)
            p1_hp_by_turn.append(p1_hp_this_turn)
            p2_hp_by_turn.append(p2_hp_this_turn)
            hp_diff = p1_hp_this_turn - p2_hp_this_turn
            #calculate the current HP difference and update the max lead achieved by either player up to that point
            if hp_diff > p1_max_lead:
                p1_max_lead = hp_diff
            if -hp_diff > p2_max_lead:
                p2_max_lead = -hp_diff
            #increment the total turn
            aggr['turns'] += 1
        #comeback: is detected if one player achieved an HP lead(50% HP advantage) and if the opposite player won KO race
        if p2_max_lead > 0.5 and aggr['p1_ko'] > aggr['p2_ko'] :
            comeback_detected = True
        elif p1_max_lead > 0.5 and aggr['p2_ko'] > aggr['p1_ko']:
            comeback_detected = True
        #*----------------------------------------------FEATURES---------------------------------------------------------
        for k, v in aggr.items():
            features[f'battle_{k}'] = v
        features.pop('battle_turns', None)
        if aggr['turns'] > 0:
            #average amount of damage dealt by each players per battle turn
            features['p1_dmg_per_turn'] = aggr['p1_dmg'] / aggr['turns']
            features['p2_dmg_per_turn'] = aggr['p2_dmg'] / aggr['turns']
            #relative damage advantage of P1 over P2. A value grater than 1 indicates that P1 dealt more total damage than P2
            features['damage_relative_ratio'] = (aggr['p1_dmg'] + 0.01) / (aggr['p2_dmg'] + 0.01)
        #total balance of damage inflicted by P1 compared to damage by P2 over the entire battle; positive value --> P1 inflicted more total dmg than P2 
        features['net_balance_damage'] = aggr['p1_dmg'] - aggr['p2_dmg']
        #total balance of knocked out Pok achieved by P1 compared to those of P2
        features['net_balance_ko'] = aggr['p1_ko'] - aggr['p2_ko']
        if len(p1_damage_by_turn) > 1:
            #mean damage per turn divided by the StaDev of damae epr turn
            p1_dmg_mean = sum(p1_damage_by_turn) / len(p1_damage_by_turn)
            p1_dmg_variance = sum((x - p1_dmg_mean)**2 for x in p1_damage_by_turn) / len(p1_damage_by_turn)
            features['p1_damage_staDev_consistency'] = p1_dmg_mean / (p1_dmg_variance**0.5 + 0.01)
        else:
            features['p1_damage_staDev_consistency'] = 0
        if len(p2_damage_by_turn) > 1:
            p2_dmg_mean = sum(p2_damage_by_turn) / len(p2_damage_by_turn)
            p2_dmg_variance = sum((x - p2_dmg_mean)**2 for x in p2_damage_by_turn) / len(p2_damage_by_turn)
            features['p2_damage_staDev_consistency'] = p2_dmg_mean / (p2_dmg_variance**0.5 + 0.01)
        else:
            features['p2_damage_staDev_consistency'] = 0
        #gap of P2's consistency score from P1's.
        features['consistency_diff'] = features['p1_damage_staDev_consistency'] - features['p2_damage_staDev_consistency']
        #divide the count of effective moves (that dealt > 25%) by the total offensive moves used
        p1_move_effectiveness = p1_effective_moves / (p1_total_moves + 1)
        features['p2_move_effect'] = p2_effective_moves / (p2_total_moves + 1)
        #capture the realtive advantage. A positive value indicates that P1 was on average more efficient with their offensive moves
        features['effectiveness_diff'] = p1_move_effectiveness - features['p2_move_effect']
        #binary feature indicating which player was the first to inflict significant damage > 15% 
        features['first_blood_p1'] = 1 if first_blood == 'p1' else 0
        #measure diversity of Pok used by P1 relative to the total size of their team
        features['p1_team_diversity'] = len(p1_pokemon_seen) / (len(p1_team) + 1)
        #measure the absolute number of P2's Pok that were actively brought
        features['p2_team_absolute'] = len(p2_pokemon_seen)
        #Binary fag indicating if the winner overcame a significant HP lead held by the opponent earlier in the match
        features['comeback'] = 1 if comeback_detected else 0
        #largest HP advantage achieved by either player at any single point in the battle
        features['max_lead_achieved'] = max(p1_max_lead, p2_max_lead)
        #KO streak analysis: analyze the KO sequence to calculate the current consecutive streak for the player who scored the KO. 
        if len(ko_sequence) > 0:
            max_streak_p1 = 0
            max_streak_p2 = 0
            current_streak = 1
            for i in range(1, len(ko_sequence)):
                if ko_sequence[i] == ko_sequence[i-1]:
                    current_streak += 1
                else:
                    if ko_sequence[i-1] == 'p1':
                        max_streak_p1 = max(max_streak_p1, current_streak)
                    else:
                        max_streak_p2 = max(max_streak_p2, current_streak)
                    current_streak = 1
            if ko_sequence[-1] == 'p1':
                max_streak_p1 = max(max_streak_p1, current_streak)
            else:
                max_streak_p2 = max(max_streak_p2, current_streak)
            #max n of consecutive KOs achieved by each player
            features['max_consKo_streak_p1'] = max_streak_p1
            features['max_consKo_streak_p2'] = max_streak_p2
            features['ko_cons_diff'] = max_streak_p1 - max_streak_p2
        else:
            features['max_consKo_streak_p1'] = 0
            features['max_consKo_streak_p2'] = 0
            features['ko_cons_diff'] = 0
        #number of the first ko or 999 if no KO occured
        features['first_ko_per_turn'] = first_ko_turn if first_ko_turn else 999
        #1 if the first KO was scored by P1
        features['first_ko_scored_by_p1'] = 1 if first_ko_by == 'p1' else 0
        features['early_first_general_ko'] = 1 if first_ko_turn and first_ko_turn <= 5 else 0
        #gap in the total number of times players switched Pok
        features['switch_diff_p1_p2'] = p1_switch_count - p2_switch_count
        #PS's switch count normalized by the total n of turns
        features['p2_switch_normalized'] = p2_switch_count / (aggr['turns'] + 1)
        #percentage of turns P1/P2's active pok was afflicted by a major status condition
        features['p1_afflicted_by_major_status'] = p1_status_turns / (aggr['turns'] + 1)
        features['p2_afflicted_by_major_status'] = p2_status_turns / (aggr['turns'] + 1)
        #gap that measures which player was more effective at avoiding status or inflicting it on the opponent
        features['status_avoiding'] = features['p2_afflicted_by_major_status'] - features['p1_afflicted_by_major_status']
        p1_lead = p1_team[0] if p1_team else {}
        p1_lead_speed = p1_lead.get('base_spe', 0)
        p2_lead_speed = p2_lead.get('base_spe', 0)
        #absolute gap in speed between the two pok at the start of the battle
        features['speed_diff'] = p1_lead_speed - p2_lead_speed
        #true if p1 is faster
        features['best_speed_advantage'] = 1 if p1_lead_speed > p2_lead_speed else 0
        features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])
        feature_list.append(features)
    return pd.DataFrame(feature_list).fillna(0)


print("\n*** Cleaning training and test data ***")
test_data = []
with open(test_file_path, 'r') as f:
    for line in f:
        test_data.append(json.loads(line))
test_data = clean_battles(test_data)

print("*** Processing training data ***")
train_data = clean_battles(train_data)
train_df = create_battle_features(train_data)

print("\n*** Processing test data ***")
test_df = create_battle_features(test_data)

features = [col for col in train_df.columns if col not in ['battle_id', 'player_won']]
X = train_df[features]
y = train_df['player_won']
X_test = test_df[features]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=42
)
print(f"Training set: {X_train.shape}, Validation set: {X_valid.shape}")

pipeline = make_pipeline(
    StandardScaler(),
    XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_estimators=200
    )
)

param_grid = {
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__max_depth': [3, 4, 5],
    'xgbclassifier__subsample': [0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.8, 1.0],
    'xgbclassifier__reg_lambda': [1, 5, 10]
}

grid_xgb = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=-1,
    cv=5,
    refit=True,
    return_train_score=True,
    verbose=1
)

print("\n*** Starting GridSearchCV 5-fold with XGBoost ***")
grid_xgb.fit(X_train, y_train)
best_params = grid_xgb.best_params_
best_score = grid_xgb.best_score_
print("\n*** Best iperparameters found ***")
pprint(best_params)
print(f"*** Best ROC-AUC CV: {best_score:.4f} ***")

best_model = grid_xgb.best_estimator_
predictions = best_model.predict(X_valid)
predictions_proba = best_model.predict_proba(X_valid)[:, 1]

print("\n*** Performance on validation set ***")
print("1. Accuracy:", round(accuracy_score(y_valid, predictions), 4))
print("2. ROC-AUC:", round(roc_auc_score(y_valid, predictions_proba), 4))
print("3. Confusion Matrix:\n", confusion_matrix(y_valid, predictions))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=kf, scoring='accuracy')

print("\n*** 5-Fold Cross-Validation (accuracy) ***")
print("1. Fold scores:", np.round(cv_scores, 4))
print("2. Media:", np.round(cv_scores.mean(), 4), "±", np.round(cv_scores.std(), 4))

best_model.fit(X, y)
test_predictions = best_model.predict(X_test)

submission_df = pd.DataFrame({
    'battle_id': test_df['battle_id'],
    'player_won': test_predictions
})
submission_df.to_csv('submission.csv', index=False)
print("\n*** File 'submission.csv' built ***")

def show_important_features_xgb(model_pipeline, feature_names: list[str]):
    try:
        xgb_model = model_pipeline.named_steps['xgbclassifier']
    except KeyError:
        print("Error: None step 'xgbclassifier' in pipeline.")
        print(model_pipeline.named_steps.keys())
        return None
    importance = xgb_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': importance,         # treat as “coefficients” (all positive)
        'Importanza_Abs': np.abs(importance)
    }).sort_values(by='Importanza_Abs', ascending=False).reset_index(drop=True)
    return importance_df

print(show_important_features_xgb(best_model, features))

def plot_feature_importance():
    plot_df = show_important_features_xgb(best_model, features) 
    plot_df['Abs_Coefficient'] = plot_df['Coefficient'].abs()
    plot_df = plot_df.sort_values(by='Abs_Coefficient', ascending=True)
    colors = ['#C44E52' if c < 0 else '#55A868' for c in plot_df['Coefficient']]
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(12, 10))
    plt.barh(
        plot_df['Feature'], 
        plot_df['Coefficient'], 
        color=colors,
        edgecolor='none',
        alpha=0.8
    )
    plt.title(
        "Features' Importance", 
        fontsize=18, 
        fontweight='bold', 
        color='black'
    )
    plt.axvline(0, color='gray', linestyle='-', linewidth=0.7)     
    plt.xlabel('Coefficient', fontsize=14, color='dimgray')
    plt.ylabel(None)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().yaxis.grid(False) 
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predict 0 (Loser)', 'Predict 1 (Winner)'],
                yticklabels=['Real 0 (Loser)', 'Real 1 (Winner)'])
    plt.title('Confusion Matrix')
    plt.ylabel('Real Class')
    plt.xlabel('Predict class')
    plt.show()

plot_confusion_matrix(y_valid, predictions)
importance_df = show_important_features_xgb(best_model, features)
plot_feature_importance()
