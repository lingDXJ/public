# improve.py
import json
import re
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import ClassifierChain
from imblearn.over_sampling import RandomOverSampler   # 仅支持单标签
import torch
from transformers import BertTokenizer, BertModel

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# 1. 数据读取 & 清洗（同原脚本）
# ----------------------------------------------------------
def clean_dict_keys(raw):
    return [{k.replace("\ufeff", "").strip(): v for k, v in item.items()} for item in raw]

with open("combined.json", encoding="utf-8-sig") as f:
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

# ----------------------------------------------------------
# 2. 逐标签独立过采样（long-form + pivot）
# ----------------------------------------------------------
# 2-1 拉成长表
long = []
for col in label_cols:
    tmp = df[["text"]].copy()
    tmp["label_name"] = col
    tmp["label_value"] = df[col]
    long.append(tmp[["text", "label_name", "label_value"]])
long_df = pd.concat(long, ignore_index=True)

# 2-2 过采样：RandomOverSampler 只认识 X, y
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(
    long_df[["text", "label_name"]],   # X
    long_df["label_value"]             # y
)

# 2-3 把标签值拼回去
X_res["label_value"] = y_res

# 2-4 pivot 回宽表
df_wide = (
    X_res
    .pivot_table(index="text", columns="label_name", values="label_value", fill_value=0)
    .reset_index()
)
X_text_res = df_wide["text"]
y_res = df_wide[label_cols]
# ----------------------------------------------------------
# 3. 轻量文本增强
# ----------------------------------------------------------
def simple_augment(text):
    reps = [("好吃", "很香"), ("便宜", "实惠"), ("包装", "袋子")]
    for old, new in reps:
        if old in text and random.random() < 0.3:
            text = text.replace(old, new)
    return text

X_text_res = X_text_res.apply(simple_augment)

# ----------------------------------------------------------
# 4. 训练 / 测试划分
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_text_res, y_res, test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# 5. BERT 特征提取
# ----------------------------------------------------------
tokenizer  = BertTokenizer.from_pretrained("bert-base-chinese")
bert_model = BertModel.from_pretrained("bert-base-chinese")

@torch.no_grad()
def bert_features(texts):
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    return bert_model(**inputs).pooler_output.numpy()

X_train_vec = bert_features(X_train.tolist())
X_test_vec  = bert_features(X_test.tolist())

# ----------------------------------------------------------
# 6. 模型：ClassifierChain + GBDT（更多树 + 阈值后移）
# ----------------------------------------------------------
base_clf = GradientBoostingClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
chain = ClassifierChain(base_clf, order='random', random_state=42)
chain.fit(X_train_vec, y_train)

# ----------------------------------------------------------
# 7. 预测 & 阈值后移（0.3）
# ----------------------------------------------------------
y_pred_prob = chain.predict_proba(X_test_vec)
y_pred = (y_pred_prob > 0.3).astype(int)

# ----------------------------------------------------------
# 8. 评估
# ----------------------------------------------------------
print("\n========== 改进后 GBDT+Chain+采样+阈值0.3 ==========")
print(classification_report(y_test, y_pred, target_names=label_cols, zero_division=0))

# ----------------------------------------------------------
# 9. 与原模型 F1 对比可视化
# ----------------------------------------------------------
def macro_f1_per_label(y_true, y_pred):
    return [f1_score(y_true.iloc[:, i], y_pred[:, i], zero_division=0)
            for i in range(y_true.shape[1])]

f1_old = [0.80, 0.09, 0.39, 0.24, 0.10, 0.00, 0.21, 0.00, 0.00]  # 原结果
f1_new = macro_f1_per_label(y_test, y_pred)

plt.figure(figsize=(10, 4))
sns.barplot(x=label_cols, y=[f1_new[i] - f1_old[i] for i in range(9)], palette="coolwarm")
plt.axhline(0, color='black', lw=0.8)
plt.title("F1 提升（改进后 - 原模型）")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Δ F1")
plt.tight_layout()
plt.show()