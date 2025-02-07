import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoConfig
from typing import Dict
from transformers import BertModel, BertConfig


############################################################
# 1. BERT の内部から q, k, v を取得するためのカスタムクラス
############################################################
class BertModelWithQKV(BertModel):
    """
    BertModel を継承し、forward hook で各レイヤーの q, k, v を保存するクラス。
    model.encoder.layer[i].attention.self にフックを仕込む。
    """

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.q_layers: Dict[int, torch.Tensor] = {}
        self.k_layers: Dict[int, torch.Tensor] = {}
        self.v_layers: Dict[int, torch.Tensor] = {}

        for layer_idx in range(config.num_hidden_layers):
            self.encoder.layer[layer_idx].attention.self.register_forward_hook(
                self._get_qkv_hook(layer_idx)
            )

    def _get_qkv_hook(self, layer_idx: int):
        def hook(module, module_input, module_output):
            hidden_states = module_input[0]  # shape: (batch_size, seq_len, hidden_dim)
            query_layer = module.query(hidden_states)
            key_layer   = module.key(hidden_states)
            value_layer = module.value(hidden_states)
            # CPU に移して保存
            self.q_layers[layer_idx] = query_layer.detach().cpu()
            self.k_layers[layer_idx] = key_layer.detach().cpu()
            self.v_layers[layer_idx] = value_layer.detach().cpu()
        return hook

    def get_qkv_from_layer(self, layer_idx: int):
        q = self.q_layers.get(layer_idx, None)
        k = self.k_layers.get(layer_idx, None)
        v = self.v_layers.get(layer_idx, None)
        return q, k, v

############################################################
# 2. モデル・トークナイザーのロード
############################################################
@st.cache_resource
def load_model_and_tokenizer(model_name="bert-base-uncased"):
    config = AutoConfig.from_pretrained(model_name)
    # カスタムモデルを使う
    model = BertModelWithQKV.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer, model

############################################################
# 3. ヒートマップを描画する関数
############################################################
def plot_heatmap(matrix, tokens, title=""):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=tokens, yticklabels=tokens, cmap="Blues", square=True)
    plt.title(title)
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()


############################################################
# 4. Streamlit アプリ本体
############################################################
st.title("BERT Pre-Softmax Attention (QK^T) Visualization")
st.sidebar.header("Model Configuration")

# モデル名の選択
model_name = st.sidebar.selectbox(
    "Choose a model:",
    ["bert-base-uncased", "bert-large-uncased"]
)

# モデルとトークナイザーをロード（カスタムモデル）
tokenizer, model = load_model_and_tokenizer(model_name)

# テキスト入力
input_text = st.text_area("Enter your text:", "She was a teacher for forty years and her writing has appeared in journals and anthologies since the early 1980s.")
if not input_text.strip():
    st.warning("Please enter some text.")
    st.stop()

# トークナイズ
inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)

# 推論実行（hooksでQ,K,Vが格納される）
with torch.no_grad():
    _ = model(**inputs)  # 出力は使わないが、内部でフックを通じ Q,K,V が保存される

# 層 (layer_idx) とヘッド (head_idx) を選択
num_layers = model.config.num_hidden_layers
# BERT のデフォルトヘッド数は 12 (bert-base), 16 (bert-large) など
num_heads = model.config.num_attention_heads

layer_idx = st.sidebar.slider("Layer:", 1, num_layers, 1)
head_idx = st.sidebar.slider("Head:", 1, num_heads, 1)

# 指定したレイヤーから Q, K を取得
# 注意: Streamlit のスライダーは 1 始まりなので、実際の内部レイヤーは `layer_idx-1`
q, k, _ = model.get_qkv_from_layer(layer_idx - 1)

# Q, K の shape は [batch_size, seq_len, hidden_dim]
# BERTSelfAttention 内ではヘッドごとに reshape / transpose されるので自前で行う
batch_size, seq_len, hidden_dim = q.shape

# num_heads と attention_head_size への分割
# hidden_dim = num_heads * head_dim となっているはず
head_dim = hidden_dim // num_heads

# [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
q = q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
k = k.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

# 選択したヘッドだけ取り出す: shape [batch_size, seq_len, head_dim]
q_head = q[:, head_idx - 1, :, :]  # (1, seq_len, head_dim)
k_head = k[:, head_idx - 1, :, :]  # (1, seq_len, head_dim)

# 1つ目のバッチのみを可視化対象とする
q_head = q_head[0]  # shape [seq_len, head_dim]
k_head = k_head[0]  # shape [seq_len, head_dim]

# ソフトマックス前のスコア (QK^T). 
# 必要なら / sqrt(head_dim) をかければ実際のAttention Scoreに近い値になる
pre_softmax_scores = torch.matmul(q_head, k_head.transpose(-1, -2))  # shape [seq_len, seq_len]

# 可視化用に numpy array へ
pre_softmax_scores = pre_softmax_scores.detach().numpy()

# トークン取得
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

if st.button("Visualize QK^T"):
    title = f"Pre-Softmax Scores (Layer {layer_idx}, Head {head_idx})"
    plot_heatmap(pre_softmax_scores, tokens, title=title)
