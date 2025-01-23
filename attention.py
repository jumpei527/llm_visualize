import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel

# モデルとトークナイザーのロード
@st.cache_resource
def load_model_and_tokenizer(model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    return tokenizer, model

# Attentionをヒートマップで表示
def plot_heatmap(attention_matrix, tokens, layer_idx, head_idx):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap="Blues", square=True)
    plt.title(f"Layer {layer_idx} - Head {head_idx} Attention Heatmap")
    plt.xlabel("Query Tokens")
    plt.ylabel("Key Tokens")
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(plt)

# Attentionの接続可視化を作成
def plot_token_interactions(attention_matrix, tokens):
    plt.figure(figsize=(8, 8))
    plt.axis('off')  # 軸を非表示にする

    # トークンを描画
    for i, token in enumerate(tokens):
        plt.text(-1, len(tokens) - i - 1, token, fontsize=10, ha='right')
        plt.text(1, len(tokens) - i - 1, token, fontsize=10, ha='left')

    # Attentionスコアを線として描画
    seq_len = len(tokens)
    for i in range(seq_len):
        for j in range(seq_len):
            weight = attention_matrix[i, j]
            if weight > 0.1:  # 閾値で調整
                alpha = min(weight, 1.0)  # 線の透明度をスコアに応じて調整
                plt.plot([-1, 1], [len(tokens) - i - 1, len(tokens) - j - 1],
                         color='blue', alpha=alpha, linewidth=weight * 2)

    st.pyplot(plt)

# [CLS] トークンとのAttentionに基づき文章をハイライト
def highlight_text_with_attention(attention_matrix, tokens):
    cls_attention = attention_matrix[0]  # [CLS]トークンとのAttention
    max_weight = max(cls_attention)  # 最大値を取得して正規化用に利用

    # トークンごとに色付け
    highlighted_text = ""
    for token, weight in zip(tokens, cls_attention):
        normalized_weight = weight / max_weight  # 正規化
        color = f"rgba(255, 255, 0, {normalized_weight})"  # 蛍光色（黄色）に透明度を適用
        highlighted_text += f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;">{token}</span> '

    # 表示
    st.markdown(f"<div style='line-height: 1.8;'>{highlighted_text}</div>", unsafe_allow_html=True)

# Streamlit UI
st.title("BERT Token Interaction Visualization")
st.sidebar.header("Model Configuration")

# モデルとトークナイザーをロード
model_name = st.sidebar.selectbox("Choose a model:", ["bert-base-uncased", "bert-large-uncased"])
tokenizer, model = load_model_and_tokenizer(model_name)

# 入力テキスト
input_text = st.text_area("Enter your text:", "She was a teacher for forty years and her writing has appeared in journals and anthologies since the early 1980s.")
if input_text.strip() == "":
    st.warning("Please enter some text to visualize.")
    st.stop()

# トークナイズ
inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)

# モデル実行
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions

# 層とヘッドの選択
num_layers = len(attentions)
num_heads = attentions[0].shape[1]
layer_idx = st.sidebar.slider("Layer:", 1, num_layers, 1)
head_idx = st.sidebar.slider("Head:", 1, num_heads, 1)

# Attention Weightsを取得
attention_matrix = attentions[layer_idx-1][0, head_idx-1].detach().numpy()
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# 可視化
if st.button("Visualize"):
    st.subheader("Attention Heatmap")
    plot_heatmap(attention_matrix, tokens, layer_idx, head_idx)
    st.subheader("Token Interaction Visualization")
    plot_token_interactions(attention_matrix, tokens)
    st.subheader("[CLS] Token Attention Highlighting")
    highlight_text_with_attention(attention_matrix, tokens)
