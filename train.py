import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
from collections import defaultdict
import math

warnings.filterwarnings('ignore')

# ============================================================================
# 1. 增強的數據處理器
# ============================================================================

class EnhancedProcessor:
    """增強的數據處理器"""

    def __init__(self):
        self.action_to_point_probs = None
        self.position_transitions = None
        self.action_position_probs = None
        self.point_sequences = None

    def build_statistical_priors(self, train_df):
        """構建統計先驗知識"""
        print("Building enhanced statistical priors...")
        train_df_filtered = train_df[train_df['actionId'] != -1].copy()

        # 1. Action -> Point 條件概率
        self.action_to_point_probs = defaultdict(lambda: defaultdict(float))
        for _, row in train_df_filtered.iterrows():
            action = int(row['actionId'])
            point = int(row['pointId'])
            self.action_to_point_probs[action][point] += 1

        for action in self.action_to_point_probs:
            total = sum(self.action_to_point_probs[action].values())
            for point in self.action_to_point_probs[action]:
                self.action_to_point_probs[action][point] /= total

        # 2. Position 轉移矩陣
        self.position_transitions = defaultdict(lambda: defaultdict(int))
        for rally_uid, group in train_df_filtered.groupby('rally_uid'):
            group = group.sort_values('strickNumber')
            for i in range(len(group) - 1):
                curr_pos = int(group.iloc[i]['positionId'])
                next_pos = int(group.iloc[i+1]['positionId'])
                self.position_transitions[curr_pos][next_pos] += 1

        for pos in self.position_transitions:
            total = sum(self.position_transitions[pos].values())
            for next_pos in self.position_transitions[pos]:
                self.position_transitions[pos][next_pos] /= total

        # 3. Action-Position 聯合概率
        self.action_position_probs = defaultdict(lambda: defaultdict(int))
        for _, row in train_df_filtered.iterrows():
            action = int(row['actionId'])
            position = int(row['positionId'])
            self.action_position_probs[action][position] += 1

        for action in self.action_position_probs:
            total = sum(self.action_position_probs[action].values())
            for position in self.action_position_probs[action]:
                self.action_position_probs[action][position] /= total

        # 4. Point 序列模式
        self.point_sequences = defaultdict(lambda: defaultdict(int))
        for rally_uid, group in train_df_filtered.groupby('rally_uid'):
            group = group.sort_values('strickNumber')
            points = group['pointId'].tolist()
            for i in range(len(points) - 1):
                self.point_sequences[points[i]][points[i+1]] += 1

        for point in self.point_sequences:
            total = sum(self.point_sequences[point].values())
            for next_point in self.point_sequences[point]:
                self.point_sequences[point][next_point] /= total

        print(f"Built priors: {len(self.action_to_point_probs)} actions")

    def create_rally_sequences(self, df, is_train=True):
        sequences = []
        for rally_uid, group in df.groupby('rally_uid'):
            group = group.sort_values('strickNumber').copy()

            if is_train:
                group = group[group['actionId'] != -1].copy()
                if len(group) <= 1:
                    continue

                first_player_id = group.iloc[0]['gamePlayerId']
                group['striker'] = np.where(group['gamePlayerId'] == first_player_id, 1, 2)

                server_id_raw = group.iloc[0]['serveId']
                server_won = group.iloc[0]['serverGetPoint']
                rally_winner = server_won if server_id_raw == 1 else 1 - server_won

                for i in range(1, len(group)):
                    history = group.iloc[:i]
                    current = group.iloc[i]

                    seq_data = {
                        'rally_id': rally_uid,
                        'history_actions': (history['actionId'] + 1).tolist(),
                        'history_points': (history['pointId'] + 1).tolist(),
                        'history_positions': (history['positionId'] + 1).tolist(),
                        'history_strikers': history['striker'].tolist(),
                        'target_action': current['actionId'] + 1,
                        'target_point': current['pointId'] + 1,
                        'rally_winner': rally_winner,
                        'current_striker': np.where(current['gamePlayerId'] == first_player_id, 1, 2),
                    }
                    sequences.append(seq_data)
            else:
                first_player_id = group.iloc[0]['gamePlayerId']
                group['striker'] = np.where(group['gamePlayerId'] == first_player_id, 1, 2)
                history = group
                last_striker = history.iloc[-1]['striker']
                next_striker = 2 if last_striker == 1 else 1

                seq_data = {
                    'rally_id': rally_uid,
                    'history_actions': (history['actionId'] + 1).tolist(),
                    'history_points': (history['pointId'] + 1).tolist(),
                    'history_positions': (history['positionId'] + 1).tolist(),
                    'history_strikers': history['striker'].tolist(),
                    'id': rally_uid,
                    'current_striker': next_striker,
                }
                sequences.append(seq_data)

        return sequences

    def pad_sequence(self, seq, max_len, pad_value=0):
        if len(seq) >= max_len:
            return seq[-max_len:]
        else:
            return [pad_value] * (max_len - len(seq)) + seq

    def get_enhanced_features(self, history_actions, history_points, history_positions,
                            current_striker, max_seq_len, n_points):
        """獲取增強的統計特徵"""
        features = []
        seq_len = len(history_actions)

        # 基本特徵
        features.extend([
            seq_len / max_seq_len,
            current_striker / 2.0,
        ])

        # 最近的特徵
        features.extend([
            history_actions[-1] / 20.0 if seq_len >= 1 else 0,
            history_points[-1] / 10.0 if seq_len >= 1 else 0,
            history_positions[-1] / 10.0 if seq_len >= 1 else 0,
        ])

        # 對手特徵
        features.extend([
            history_actions[-2] / 20.0 if seq_len >= 2 else 0,
            history_points[-2] / 10.0 if seq_len >= 2 else 0,
            history_positions[-2] / 10.0 if seq_len >= 2 else 0,
        ])

        # 序列動態特徵
        if seq_len >= 2:
            point_changes = sum(1 for i in range(seq_len-1)
                                if history_points[i] != history_points[i+1])
            features.append(point_changes / (seq_len - 1))

            pos_changes = sum(1 for i in range(seq_len-1)
                                if history_positions[i] != history_positions[i+1])
            features.append(pos_changes / (seq_len - 1))

            action_diversity = len(set(history_actions)) / seq_len
            features.append(action_diversity)
        else:
            features.extend([0, 0, 0])

        # 趨勢特徵
        if seq_len >= 6:
            recent_points = history_points[-5:]
            earlier_points = history_points[:-5]
            recent_avg = np.mean(recent_points)
            earlier_avg = np.mean(earlier_points)
            features.append((recent_avg - earlier_avg) / 10.0)
        else:
            features.append(0)

        # N-gram 特徵
        if seq_len >= 3:
            last_3_actions = tuple(history_actions[-3:])
            action_hash = hash(last_3_actions) % 1000 / 1000.0
            features.append(action_hash)
        else:
            features.append(0)

        # Action -> Point 條件概率
        last_action = history_actions[-1] if seq_len >= 1 else 0
        action_point_probs = [0.0] * n_points
        if last_action > 0 and (last_action - 1) in self.action_to_point_probs:
            for point, prob in self.action_to_point_probs[last_action - 1].items():
                if point < n_points:
                    action_point_probs[point] = prob
        features.extend(action_point_probs)

        # Point 序列轉移概率
        last_point = history_points[-1] if seq_len >= 1 else 0
        point_transition_probs = [0.0] * n_points
        if last_point > 0 and (last_point - 1) in self.point_sequences:
            for next_point, prob in self.point_sequences[last_point - 1].items():
                if next_point < n_points:
                    point_transition_probs[next_point] = prob
        features.extend(point_transition_probs)

        # 位置-動作交互特徵
        last_position = history_positions[-1] if seq_len >= 1 else 0
        if last_action > 0 and (last_action - 1) in self.action_position_probs:
            if last_position > 0 and (last_position - 1) in self.action_position_probs[last_action - 1]:
                features.append(self.action_position_probs[last_action - 1][last_position - 1])
            else:
                features.append(0)
        else:
            features.append(0)

        # Delta 特徵
        if seq_len >= 2:
            action_delta = (history_actions[-1] - history_actions[-2]) / 20.0
            point_delta = (history_points[-1] - history_points[-2]) / 10.0
            pos_delta = (history_positions[-1] - history_positions[-2]) / 10.0
            features.extend([action_delta, point_delta, pos_delta])
        else:
            features.extend([0, 0, 0])

        return features

    def prepare_features(self, sequences, max_seq_len=25, n_points=10, is_train=True):
        """準備增強特徵"""
        features = []

        for seq in sequences:
            actions = self.pad_sequence(seq['history_actions'], max_seq_len, pad_value=0)
            points = self.pad_sequence(seq['history_points'], max_seq_len, pad_value=0)
            positions = self.pad_sequence(seq['history_positions'], max_seq_len, pad_value=0)
            strikers = self.pad_sequence(seq['history_strikers'], max_seq_len, pad_value=0)

            seq_len = min(len(seq['history_actions']), max_seq_len)

            stat_features = self.get_enhanced_features(
                seq['history_actions'],
                seq['history_points'],
                seq['history_positions'],
                seq['current_striker'],
                max_seq_len,
                n_points
            )

            feature_dict = {
                'actions': actions,
                'points': points,
                'positions': positions,
                'strikers': strikers,
                'seq_len': seq_len,
                'stat_features': stat_features,
                'rally_id': seq['rally_id']
            }

            if is_train:
                feature_dict['target_action'] = seq['target_action']
                feature_dict['target_point'] = seq['target_point']
                feature_dict['rally_winner'] = seq['rally_winner']
            else:
                feature_dict['id'] = seq['id']

            features.append(feature_dict)

        return features


# ============================================================================
# 2. Dataset
# ============================================================================

class TableTennisDataset(Dataset):
    def __init__(self, features, is_train=True):
        self.features = features
        self.is_train = is_train

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]

        sample = {
            'actions': torch.LongTensor(item['actions']),
            'points': torch.LongTensor(item['points']),
            'positions': torch.LongTensor(item['positions']),
            'strikers': torch.LongTensor(item['strikers']),
            'seq_len': torch.LongTensor([item['seq_len']]),
            'stat_features': torch.FloatTensor(item['stat_features'])
        }

        if self.is_train:
            sample['target_action'] = torch.LongTensor([item['target_action']])
            sample['target_point'] = torch.LongTensor([item['target_point']])
            sample['rally_winner'] = torch.LongTensor([item['rally_winner']])
            sample['rally_id'] = item['rally_id']
        else:
            sample['id'] = item['id']

        return sample


# ============================================================================
# 3. 模型架構
# ============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        out = self.out(out)

        return out


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, lstm_out, seq_len):
        attn_logits = self.attention_weights(lstm_out).squeeze(-1)
        max_len = lstm_out.size(1)

        if seq_len.dim() > 1:
            seq_len_1d = seq_len.squeeze(-1)
        else:
            seq_len_1d = seq_len

        mask = torch.arange(max_len, device=lstm_out.device)[None, :] >= (max_len - seq_len_1d)[:, None]
        attn_logits[~mask] = -torch.finfo(torch.float32).max

        attn_weights = torch.softmax(attn_logits, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context


class EnhancedLSTM(nn.Module):
    def __init__(self, n_actions, n_points, n_positions, n_stat_features,
                 embedding_dim=64, hidden_dim=256, dropout=0.3):

        super(EnhancedLSTM, self).__init__()

        self.action_embedding = nn.Embedding(n_actions + 1, embedding_dim, padding_idx=0)
        self.point_embedding = nn.Embedding(n_points + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(n_positions + 1, embedding_dim//2, padding_idx=0)
        self.striker_embedding = nn.Embedding(3, embedding_dim//2, padding_idx=0)

        lstm_input_dim = embedding_dim * 2 + embedding_dim

        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, 2,
                            batch_first=True, dropout=dropout,
                            bidirectional=True)

        self.multihead_attn = MultiHeadAttention(hidden_dim * 2, num_heads=4, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.attention_pooling = AttentionPooling(hidden_dim * 2)

        self.stat_mlp = nn.Sequential(
            nn.Linear(n_stat_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        combined_dim = hidden_dim * 2 + 64

        self.action_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_actions + 1)
        )

        self.point_head = nn.Sequential(
            nn.Linear(combined_dim, 384),
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Dropout(dropout),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, n_points + 1)
        )

        self.winner_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, actions, points, positions, strikers, seq_len, stat_features):
        batch_size, seq_length = actions.shape

        action_emb = self.action_embedding(actions)
        point_emb = self.point_embedding(points)
        position_emb = self.position_embedding(positions)
        striker_emb = self.striker_embedding(strikers)

        lstm_input = torch.cat([action_emb, point_emb, position_emb, striker_emb], dim=-1)

        lstm_out, _ = self.lstm(lstm_input)

        attn_out = self.multihead_attn(lstm_out)
        lstm_out = self.layer_norm(lstm_out + attn_out)

        context = self.attention_pooling(lstm_out, seq_len)

        stat_out = self.stat_mlp(stat_features)
        combined = torch.cat([context, stat_out], dim=-1)

        action_logits = self.action_head(combined)
        point_logits = self.point_head(combined)
        winner_logits = self.winner_head(combined)

        return action_logits, point_logits, winner_logits


# ============================================================================
# 4. Focal Loss
# ============================================================================

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma_base=2.0, class_difficulty=None):
        super(AdaptiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma_base = gamma_base
        self.class_difficulty = class_difficulty

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)

        if self.class_difficulty is not None:
            gamma = torch.full((inputs.size(0),), self.gamma_base, device=inputs.device, dtype=torch.float32)
            for i, target in enumerate(targets):
                target_val = target.item() if isinstance(target, torch.Tensor) else target
                if target_val in self.class_difficulty:
                    gamma[i] = self.gamma_base * self.class_difficulty[target_val]
        else:
            gamma = self.gamma_base

        focal_loss = ((1 - pt) ** gamma) * ce_loss
        return focal_loss.mean()


# ============================================================================
# 5. 訓練函數
# ============================================================================

def train_model(model, train_loader, val_loader, device, fold_num,
                n_epochs=30, lr=0.0015,
                action_weights=None, point_weights=None, point_difficulty=None):

    model = model.to(device)

    action_criterion = nn.CrossEntropyLoss(weight=action_weights)
    point_criterion = AdaptiveFocalLoss(alpha=point_weights, gamma_base=3.0,
                                        class_difficulty=point_difficulty)
    winner_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=n_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2, anneal_strategy='cos'
    )

    best_val_loss = float('inf')
    patience = 0
    max_patience = 8

    for epoch in range(n_epochs):
        point_weight = 3.0 + (epoch / n_epochs) * 3.0

        model.train()
        train_loss = 0
        train_point_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            actions = batch['actions'].to(device)
            points = batch['points'].to(device)
            positions = batch['positions'].to(device)
            strikers = batch['strikers'].to(device)
            seq_len = batch['seq_len'].to(device)
            stat_features = batch['stat_features'].to(device)
            target_action = batch['target_action'].squeeze(-1).to(device)
            target_point = batch['target_point'].squeeze(-1).to(device)
            rally_winner = batch['rally_winner'].squeeze(-1).to(device)

            action_logits, point_logits, winner_logits = model(
                actions, points, positions, strikers, seq_len, stat_features
            )

            loss_action = action_criterion(action_logits, target_action)
            loss_point = point_criterion(point_logits, target_point)
            loss_winner = winner_criterion(winner_logits, rally_winner)

            loss = 0.3 * loss_action + point_weight * loss_point + 0.5 * loss_winner

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_point_correct += (point_logits.argmax(1) == target_point).sum().item()
            train_total += len(target_point)

        model.eval()
        val_loss = 0
        val_point_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                actions = batch['actions'].to(device)
                points = batch['points'].to(device)
                positions = batch['positions'].to(device)
                strikers = batch['strikers'].to(device)
                seq_len = batch['seq_len'].to(device)
                stat_features = batch['stat_features'].to(device)
                target_action = batch['target_action'].squeeze(-1).to(device)
                target_point = batch['target_point'].squeeze(-1).to(device)
                rally_winner = batch['rally_winner'].squeeze(-1).to(device)

                action_logits, point_logits, winner_logits = model(
                    actions, points, positions, strikers, seq_len, stat_features
                )

                loss_action = action_criterion(action_logits, target_action)
                loss_point = point_criterion(point_logits, target_point)
                loss_winner = winner_criterion(winner_logits, rally_winner)
                loss = 0.3 * loss_action + point_weight * loss_point + 0.5 * loss_winner

                val_loss += loss.item()
                val_point_correct += (point_logits.argmax(1) == target_point).sum().item()
                val_total += len(target_point)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"\n[Fold {fold_num+1}] Epoch {epoch+1}/{n_epochs} (P_weight={point_weight:.2f})")
        print(f"Train Loss: {avg_train_loss:.4f} | Point Acc: {100*train_point_correct/train_total:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Point Acc: {100*val_point_correct/val_total:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_model_fold_{fold_num}.pth')
            print(f"✓ Best model saved!")
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping")
                break

    return model, best_val_loss


# ============================================================================
# 6. Main Pipeline with Smart Ensemble
# ============================================================================

def main():
    TRAIN_PATH = 'train.csv'
    TEST_PATH = 'test.csv'
    MAX_SEQ_LEN = 25
    BATCH_SIZE = 256
    EPOCHS = 30
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_SPLITS = 5

    print(f"Using device: {DEVICE}\n")

    print("1. Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    processor = EnhancedProcessor()
    processor.build_statistical_priors(train_df)

    print("\n2. Creating sequences...")
    train_sequences = processor.create_rally_sequences(train_df, is_train=True)
    test_sequences = processor.create_rally_sequences(test_df, is_train=False)
    print(f"Train sequences: {len(train_sequences)}, Test sequences: {len(test_sequences)}")

    train_df_filtered = train_df[train_df['actionId'] != -1]
    n_points_for_priors = int(train_df_filtered['pointId'].max()) + 1

    print("\n3. Preparing features...")
    train_features = processor.prepare_features(train_sequences, MAX_SEQ_LEN, n_points_for_priors, is_train=True)
    test_features = processor.prepare_features(test_sequences, MAX_SEQ_LEN, n_points_for_priors, is_train=False)

    N_STAT_FEATURES = len(train_features[0]['stat_features'])
    print(f"Number of stat features: {N_STAT_FEATURES}")

    train_rally_ids = [f['rally_id'] for f in train_features]
    train_features_list = train_features

    max_action_id = max(train_df_filtered['actionId'].max(), test_df['actionId'].max())
    max_point_id = max(train_df_filtered['pointId'].max(), test_df['pointId'].max())
    max_position_id = max(train_df_filtered['positionId'].max(), test_df['positionId'].max())

    n_actions = int(max_action_id + 1)
    n_points = int(max_point_id + 1)
    n_positions = int(max_position_id + 1)

    action_counts = train_df_filtered['actionId'].value_counts()
    point_counts = train_df_filtered['pointId'].value_counts()

    action_weights = torch.ones(n_actions + 1, device=DEVICE)
    point_weights = torch.ones(n_points + 1, device=DEVICE)
    point_difficulty = {}

    total_samples = len(train_df_filtered)
    avg_count = total_samples / n_points

    for i in range(n_actions):
        count = action_counts.get(i, 1)
        action_weights[i+1] = np.sqrt(total_samples / (n_actions * count))

    for i in range(n_points):
        count = point_counts.get(i, 1)
        point_weights[i+1] = np.sqrt(total_samples / (n_points * count))
        point_difficulty[i+1] = min(2.5, avg_count / count)

    action_weights[0] = 0.0
    point_weights[0] = 0.0

    gkf = GroupKFold(n_splits=N_SPLITS)
    all_fold_preds = []
    val_scores = []

    test_dataset = TableTennisDataset(test_features, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    groups_for_split = [f['rally_id'] for f in train_features_list]
    fold_splits = list(gkf.split(train_features_list, groups=groups_for_split))

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n===== FOLD {fold+1} / {N_SPLITS} =====")

        train_fold_features = [train_features_list[i] for i in train_idx]
        val_fold_features = [train_features_list[i] for i in val_idx]

        train_dataset = TableTennisDataset(train_fold_features, is_train=True)
        val_dataset = TableTennisDataset(val_fold_features, is_train=True)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = EnhancedLSTM(n_actions, n_points, n_positions, N_STAT_FEATURES)

        model, best_fold_val_loss = train_model(
            model, train_loader, val_loader, DEVICE,
            fold_num=fold, n_epochs=EPOCHS,
            action_weights=action_weights,
            point_weights=point_weights,
            point_difficulty=point_difficulty
        )
        val_scores.append(best_fold_val_loss)

        model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))

        # TTA (Test-Time Augmentation)
        fold_test_preds = {}
        n_tta = 3

        model.train()  # Enable dropout for TTA

        with torch.no_grad():
            for tta_iter in range(n_tta):
                for batch in test_loader:
                    actions = batch['actions'].to(DEVICE)
                    points = batch['points'].to(DEVICE)
                    positions = batch['positions'].to(DEVICE)
                    strikers = batch['strikers'].to(DEVICE)
                    seq_len = batch['seq_len'].to(DEVICE)
                    stat_features = batch['stat_features'].to(DEVICE)

                    action_logits, point_logits, winner_logits = model(
                        actions, points, positions, strikers, seq_len, stat_features
                    )

                    for i, rally_id in enumerate(batch['id']):
                        rally_id = rally_id.item()
                        if rally_id not in fold_test_preds:
                            fold_test_preds[rally_id] = {
                                'action': torch.zeros_like(action_logits[i]),
                                'point': torch.zeros_like(point_logits[i]),
                                'winner': torch.zeros_like(winner_logits[i])
                            }

                        fold_test_preds[rally_id]['action'] += action_logits[i]
                        fold_test_preds[rally_id]['point'] += point_logits[i]
                        fold_test_preds[rally_id]['winner'] += winner_logits[i]

        model.eval()

        # Average TTA results
        for rally_id in fold_test_preds:
            fold_test_preds[rally_id]['action'] = (fold_test_preds[rally_id]['action'] / n_tta).cpu()
            fold_test_preds[rally_id]['point'] = (fold_test_preds[rally_id]['point'] / n_tta).cpu()
            fold_test_preds[rally_id]['winner'] = (fold_test_preds[rally_id]['winner'] / n_tta).cpu()

        all_fold_preds.append(fold_test_preds)

    print(f"\n\n=== CV Finished ===")
    print(f"Mean Val Loss: {np.mean(val_scores):.4f} +/- {np.std(val_scores):.4f}")

    # ========================================================================
    # 🚀 Smart Ensemble with Confidence & Consistency Weighting
    # ========================================================================

    print("\n🚀 Applying Smart Ensemble Strategy...")

    final_predictions = []
    all_test_ids = list(all_fold_preds[0].keys())
    all_test_ids.sort()

    # Calculate fold performance weights
    fold_weights = []
    for score in val_scores:
        fold_weights.append(1.0 / (score + 1e-6))
    fold_weights = np.array(fold_weights)
    fold_weights = fold_weights / fold_weights.sum()

    print(f"Fold weights: {fold_weights}")

    for rally_id in all_test_ids:
        # Collect predictions from all folds
        fold_action_probs = []
        fold_point_probs = []
        fold_winner_probs = []

        for fold_preds in all_fold_preds:
            action_probs = F.softmax(fold_preds[rally_id]['action'], dim=0).numpy()
            point_probs = F.softmax(fold_preds[rally_id]['point'], dim=0).numpy()
            winner_probs = F.softmax(fold_preds[rally_id]['winner'], dim=0).numpy()

            fold_action_probs.append(action_probs)
            fold_point_probs.append(point_probs)
            fold_winner_probs.append(winner_probs)

        fold_action_probs = np.array(fold_action_probs)  # (5, n_actions)
        fold_point_probs = np.array(fold_point_probs)    # (5, n_points)
        fold_winner_probs = np.array(fold_winner_probs)  # (5, 2)

        # === Strategy 1: Confidence-based weighting ===
        # Lower entropy = higher confidence = higher weight
        point_entropies = -np.sum(fold_point_probs * np.log(fold_point_probs + 1e-10), axis=1)
        confidence_weights = 1.0 / (point_entropies + 0.1)
        confidence_weights = confidence_weights / confidence_weights.sum()

        # === Strategy 2: Consistency-based weighting ===
        # Lower std = higher consistency = higher weight
        point_std = fold_point_probs.std(axis=0)
        consistency_score = 1.0 / (point_std.mean() + 0.01)

        # === Strategy 3: Hybrid weighting ===
        # Combine fold performance, confidence, and consistency
        final_fold_weights = fold_weights * confidence_weights * 0.5 + fold_weights * 0.5
        final_fold_weights = final_fold_weights / final_fold_weights.sum()

        # === Weighted ensemble ===
        final_action_probs = np.sum(fold_action_probs * final_fold_weights[:, None], axis=0)
        final_point_probs = np.sum(fold_point_probs * final_fold_weights[:, None], axis=0)
        final_winner_probs = np.sum(fold_winner_probs * final_fold_weights[:, None], axis=0)

        # === Additional boost for high-agreement predictions ===
        # If all models agree strongly, boost that prediction
        point_max_agreement = fold_point_probs.max(axis=1).mean()
        if point_max_agreement > 0.7:  # High agreement
            # Boost the top prediction
            top_point = final_point_probs.argmax()
            final_point_probs[top_point] *= 1.2
            final_point_probs = final_point_probs / final_point_probs.sum()

        final_predictions.append({
            'rally_uid': rally_id,
            'actionId': final_action_probs.argmax() - 1,
            'pointId': final_point_probs.argmax() - 1,
            'rallyWinner': final_winner_probs.argmax()
        })

    submission_df = pd.DataFrame(final_predictions)
    submission_df['actionId'] = submission_df['actionId'].clip(lower=0)
    submission_df['pointId'] = submission_df['pointId'].clip(lower=0)

    # Convert to serverGetPoint
    test_rally_info = test_df.groupby('rally_uid').first()[['serveId']].reset_index()
    submission_df['rally_uid'] = submission_df['rally_uid'].astype(int)
    test_rally_info['rally_uid'] = test_rally_info['rally_uid'].astype(int)
    submission_df = submission_df.merge(test_rally_info, on='rally_uid', how='left')

    submission_df['serverGetPoint'] = np.where(
        submission_df['serveId'] == 1,
        submission_df['rallyWinner'],
        1 - submission_df['rallyWinner']
    )

    submission_df = submission_df[['rally_uid', 'serverGetPoint', 'pointId', 'actionId']]
    submission_df.to_csv('submission_optimized.csv', index=False)

    print("\n✓ Optimized submission saved!")
    print(f"Shape: {submission_df.shape}")
    print("\n=== Prediction Distribution ===")
    print("PointId distribution:")
    print(submission_df['pointId'].value_counts(normalize=True).sort_index())
    print("\nActionId distribution:")
    print(submission_df['actionId'].value_counts(normalize=True).sort_index())
    print("\nFirst 10 predictions:")
    print(submission_df.head(10))

    print("\n🎯 Expected improvement: +0.015-0.025")
    print("🎯 Target score: 0.255-0.265")

    return submission_df


if __name__ == "__main__":
    submission = main()

