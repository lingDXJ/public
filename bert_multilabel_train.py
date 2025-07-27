import json
import pandas as pd
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.ensemble import GradientBoostingClassifier
from transformers import BertTokenizer, BertModel
import random
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multioutput import MultiOutputClassifier

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 0. 多标签包装器（保持不变）
# ------------------------------------------------------------------
class MultiLabelClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.model_ = None

    def fit(self, X, y):
        self.model_ = MultiOutputClassifier(self.base_estimator).fit(X, y)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model_.predict(X)

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model_.predict_proba(X)

# ------------------------------------------------------------------
# 1. 数据预处理（保持不变）
# ------------------------------------------------------------------
def clean_dict_keys(raw_data):
    return [{k.replace("\ufeff", "").strip(): v for k, v in item.items()} for item in raw_data]

with open("combined.json", "r", encoding="utf-8-sig") as f:
    data = clean_dict_keys(json.load(f))

comments, scores = [], []
for item in data:
    comment = item.get("评论内容") or item.get("comment") or ""
    score = item.get("评分") or item.get("score") or 0
    if isinstance(comment, str) and comment.strip():
        comments.append(comment.strip())
        scores.append(score)

df = pd.DataFrame({"text": comments, "score": scores})
df["score"] = pd.to_numeric(df["score"], errors="coerce").astype(float).dropna()
df["text"] = df["text"].apply(lambda x: re.sub(r"[^\w\s\u4e00-\u9fa5]", "", str(x)).lower())
df = df[df["text"].str.strip() != ""]

label_cols = [
    "positive_taste", "positive_price", "positive_package",
    "negative_taste", "negative_price", "negative_package",
    "neutral_taste", "neutral_price", "neutral_package"
]
keywords = {
    "positive_taste":  r"好吃|爱吃|喜欢吃|香|适口|炫|食欲|营养|冻干",
    "positive_price":  r"便宜|实惠|性价比|划算|优惠",
    "positive_package":r"包装|密封|外观|精美",
    "negative_taste":  r"不爱吃|不喜欢吃|挑食|不适口|腥|不香|不吃|吐|呕吐",
    "negative_price":  r"贵|价格高|价格偏贵|不划算|太贵|性价比低",
    "negative_package":r"包装差|包装烂|包装不好|破损|密封不好|外包装差",
    "neutral_taste":   r"还行|还可以|一般|可以接受|一般般|还不错",
    "neutral_price":   r"价格合理|价格还行|不贵不便宜|价格一般|刚刚好",
    "neutral_package": r"包装一般|包装还行|包装正常|包装可以接受"
}
for col, kw in keywords.items():
    df[col] = df["text"].str.contains(kw, regex=True).astype(int)

X_train, X_test, y_train, y_test, score_train, score_test = train_test_split(
    df["text"], df[label_cols], df["score"], test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 2. BERT 特征提取（保持不变）
# ------------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert_model = BertModel.from_pretrained("bert-base-chinese")

def bert_features(texts):
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.pooler_output.numpy()

X_train_vec = bert_features(X_train.tolist())
X_test_vec  = bert_features(X_test.tolist())

# ------------------------------------------------------------------
# 3. 训练 GBDT（保留 staged_predict 以供绘制曲线）
# ------------------------------------------------------------------
gbdt = GradientBoostingClassifier(
    random_state=42,
    n_estimators=800,        # 可适当调大
    learning_rate=0.04,
    max_depth=5,
    verbose=0
)
model_gbdt = MultiLabelClassifierWrapper(gbdt)
model_gbdt.fit(X_train_vec, y_train)

# ------------------------------------------------------------------
# 4. 评估（保持不变）
# ------------------------------------------------------------------
pred_labels = model_gbdt.predict(X_test_vec)
print("\n========== GBDT 多标签分类报告 ==========")
print(classification_report(y_test, pred_labels, target_names=label_cols, zero_division=0))

# ------------------------------------------------------------------
# 5. 鲁棒性评估（保持不变）
# ------------------------------------------------------------------
def perturb_text(text, ratio=0.1):
    words = list(text)
    n_drop = max(1, int(len(words) * ratio))
    drop_idx = random.sample(range(len(words)), n_drop)
    return ''.join([w for i, w in enumerate(words) if i not in drop_idx])

def synonym_replacement(text):
    return text.replace("好吃", "非常好吃").replace("便宜", "很便宜")

text_list = X_test.tolist()
base_vec   = bert_features(text_list)
pert_vec   = bert_features([perturb_text(t) for t in text_list])
adv_vec    = bert_features([synonym_replacement(t) for t in text_list])

def predict_vec(vec):
    return model_gbdt.predict(vec)

base_pred  = predict_vec(base_vec)
pert_pred  = predict_vec(pert_vec)
adv_pred   = predict_vec(adv_vec)

robust_results = []
for tag_idx, tag in enumerate(label_cols):
    y_true = y_test.iloc[:, tag_idx].values
    for scenario, pred in zip(["Base", "Perturb 10%", "Adv-Syn"], [base_pred, pert_pred, adv_pred]):
        report = classification_report(y_true, pred[:, tag_idx], output_dict=True, zero_division=0)
        f1 = report.get('1', {}).get('f1-score', 0.0)
        robust_results.append({"Tag": tag, "Scenario": scenario, "F1": f1})

robust_df = pd.DataFrame(robust_results)

# ------------------------------------------------------------------
# 6. 可视化（保持不变）
# ------------------------------------------------------------------
plt.figure(figsize=(10,4))
sns.barplot(x=label_cols, y=df[label_cols].sum().values, color='steelblue')
plt.title("Tail-Class Distribution")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(18,6))
sns.barplot(x="Tag", y="F1", hue="Scenario", data=robust_df, palette="Set2")
plt.title("Robustness of All 9 Labels (GBDT)")
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

pt_idx = label_cols.index("positive_taste")
cm = confusion_matrix(y_test.iloc[:, pt_idx], base_pred[:, pt_idx])
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - positive_taste (GBDT)")
plt.xlabel("Pred"); plt.ylabel("True")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 7. 训练 / 验证 loss 曲线（新增）
# ------------------------------------------------------------------
# 对每个标签分别计算 log-loss，然后求平均
n_estimators = gbdt.n_estimators
train_losses = np.zeros((n_estimators, len(label_cols)))
val_losses   = np.zeros((n_estimators, len(label_cols)))

# 用原生模型对每个标签单独训练，以便获得 staged_decision_function
for idx, tag in enumerate(label_cols):
    y_tr = y_train.iloc[:, idx]
    y_te = y_test.iloc[:, idx]

    base_clf = GradientBoostingClassifier(
        random_state=42,
        n_estimators=n_estimators,
        learning_rate=0.01,
        max_depth=3,
        verbose=0
    ).fit(X_train_vec, y_tr)

    # staged_decision_function 返回的是 decision_function 值（未经过 sigmoid）
    for stage, y_pred_decision in enumerate(base_clf.staged_decision_function(X_train_vec)):
        # sigmoid 得到概率
        y_pred_prob = 1 / (1 + np.exp(-y_pred_decision))
        train_losses[stage, idx] = log_loss(y_tr, y_pred_prob, labels=[0, 1])

    for stage, y_pred_decision in enumerate(base_clf.staged_decision_function(X_test_vec)):
        y_pred_prob = 1 / (1 + np.exp(-y_pred_decision))
        val_losses[stage, idx] = log_loss(y_te, y_pred_prob, labels=[0, 1])

# 取 9 个标签的平均
train_loss_mean = train_losses.mean(axis=1)
val_loss_mean   = val_losses.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(range(1, n_estimators+1), train_loss_mean, label="Train log-loss")
plt.plot(range(1, n_estimators+1), val_loss_mean, label="Val log-loss")
plt.xlabel("Boosting Iteration")
plt.ylabel("Mean log-loss (9 labels)")
plt.title("GBDT Training / Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("BERT + GBDT 融合评估 & 训练曲线绘制完成！")