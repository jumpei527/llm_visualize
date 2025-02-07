import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List
from transformers import BertModel, BertConfig, BertTokenizer
from sklearn.decomposition import PCA
import umap


###############################################################################
# 1. BERT の内部から q, k, v を取得するためのカスタムクラス
###############################################################################
class BertModelWithQKV(BertModel):
    """
    BertModel を継承し、forward hook で各レイヤーの q, k, v を保存するクラス。
    model.encoder.layer[i].attention.self にフックを仕込む。
    """

    def __init__(self, config: BertConfig):
        super().__init__(config)

        # レイヤーごとの q, k, v を保持する辞書 (layer_idx -> Tensor)
        self.q_layers: Dict[int, torch.Tensor] = {}
        self.k_layers: Dict[int, torch.Tensor] = {}
        self.v_layers: Dict[int, torch.Tensor] = {}

        # BertEncoder の各レイヤーにフックを登録
        for layer_idx in range(config.num_hidden_layers):
            self.encoder.layer[layer_idx].attention.self.register_forward_hook(
                self._get_qkv_hook(layer_idx)
            )

    def _get_qkv_hook(self, layer_idx: int):
        """
        指定レイヤーの BertSelfAttention から、query/key/value 計算結果をフックで取り出す。
        """
        def hook(module, module_input, module_output):
            # module_input: (hidden_states, ) のタプル
            hidden_states = module_input[0]  # shape: (batch_size, seq_len, hidden_dim)

            # BertSelfAttention 内部実装:
            # query_layer = self.query(hidden_states)
            # key_layer   = self.key(hidden_states)
            # value_layer = self.value(hidden_states)
            query_layer = module.query(hidden_states)
            key_layer   = module.key(hidden_states)
            value_layer = module.value(hidden_states)

            # CPU に移して保持しておく（GPU でもよければ detach のみでもOK）
            self.q_layers[layer_idx] = query_layer.detach().cpu()
            self.k_layers[layer_idx] = key_layer.detach().cpu()
            self.v_layers[layer_idx] = value_layer.detach().cpu()
        return hook

    def get_qkv_from_layer(self, layer_idx: int):
        """
        フォワード後にフックで格納された q, k, v を返す。
        """
        q = self.q_layers.get(layer_idx, None)
        k = self.k_layers.get(layer_idx, None)
        v = self.v_layers.get(layer_idx, None)
        return q, k, v


###############################################################################
# 2. モデル読み込み & テキスト前処理ユーティリティ
###############################################################################
def load_bert_with_qkv(model_name="bert-base-uncased") -> BertModelWithQKV:
    """
    HuggingFace の "bert-base-uncased" 等を元にカスタム BertModelWithQKV を作る。
    """
    config = BertConfig.from_pretrained(model_name, output_attentions=True)
    model = BertModelWithQKV.from_pretrained(model_name, config=config)
    return model

def preprocess_text(text: str, tokenizer, max_length=128):
    """
    BERT 用の入力 IDs, attention masks 等を作る。
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    return inputs

def split_heads(tensor: torch.Tensor, num_heads: int):
    """
    BERT の (batch_size, seq_len, hidden_dim) を
    (batch_size, num_heads, seq_len, head_dim) に reshape する。
    """
    batch_size, seq_len, hidden_dim = tensor.shape
    head_dim = hidden_dim // num_heads
    # (batch_size, seq_len, num_heads, head_dim)
    tensor = tensor.view(batch_size, seq_len, num_heads, head_dim)
    # (batch_size, num_heads, seq_len, head_dim)
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    return tensor


###############################################################################
# 3. ヘッドごとに UMAP/PCA モデルをフィッティング
###############################################################################
def fit_umap_or_pca_per_head(
    texts: List[str],
    tokenizer,
    model: BertModelWithQKV,
    num_layers: int = 12,
    num_heads: int = 12,
    use_umap: bool = True,
):
    """
    全レイヤーの v を集めて、ヘッドごとに UMAP or PCA へフィットする。
    レイヤー情報は無視し、ヘッドごとに統一された次元削減モデルを作成。
    """
    # ヘッドごとの累積データ
    head_accumulated_data = [[] for _ in range(num_heads)]

    for text in texts:
        inputs = preprocess_text(text, tokenizer)
        # フォワード -> フックが走る
        with torch.no_grad():
            model(**inputs)

        # 各レイヤーからのデータを累積
        for layer_idx in range(num_layers):
            q, k, v = model.get_qkv_from_layer(layer_idx)
            v_split = split_heads(v, num_heads=num_heads)[0]  # batch_size=1 前提

            for head_idx in range(num_heads):
                head_v = v_split[head_idx]  # shape: (seq_len, head_dim)
                head_accumulated_data[head_idx].append(head_v.detach().cpu().numpy())

    # 各ヘッドごとに concat -> UMAP or PCA fit
    reducers = []
    for head_idx in range(num_heads):
        data_head = np.concatenate(head_accumulated_data[head_idx], axis=0)  # (全トークン数, head_dim)
        if use_umap:
            reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        else:
            reducer = PCA(n_components=2)

        reducer.fit(data_head)
        reducers.append(reducer)

    print("=== Done fitting UMAP/PCA per head ===")
    return reducers


###############################################################################
# 4. 可視化用関数（ヘッドごとの軸範囲を固定）
###############################################################################
def plot_head_layer_individual_figs(
    text: str,
    tokenizer,
    model: BertModelWithQKV,
    reducers_per_head,
    num_layers: int = 12,
    num_heads: int = 12,
    max_length=20,
    output_dir: str = "qkv_outputs",
):
    """
    各ヘッド × 各レイヤーごとに図を作り、下記パスに保存:
        {output_dir}/head{head_idx}/layer{layer_idx}.png

    なお、同じヘッド内で x,y 軸を揃えたい場合は、レイヤー分の投影結果を
    先にまとめて算出して min/max を取得し、各レイヤー描画時に同じ軸範囲を指定する。
    """

    # まず対象テキストを forward し、全レイヤーの q, k, v をフックで取得
    inputs = preprocess_text(text, tokenizer, max_length=max_length)
    with torch.no_grad():
        model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    seq_len = len(tokens)

    # データを一括で保存するための辞書
    data_dict = {}
    # ヘッドごとの全レイヤーを通しての x, y を保持
    x_vals_per_head = [[] for _ in range(num_heads)]
    y_vals_per_head = [[] for _ in range(num_heads)]

    # -------------------
    # すべての (layer, head) で 2次元埋め込みと CLS-スコアを算出
    # -------------------
    for layer_idx in range(num_layers):
        q, k, v = model.get_qkv_from_layer(layer_idx)
        q_split = split_heads(q, num_heads=num_heads)[0]  # (num_heads, seq_len, head_dim)
        k_split = split_heads(k, num_heads=num_heads)[0]
        v_split = split_heads(v, num_heads=num_heads)[0]

        for head_idx in range(num_heads):
            head_q = q_split[head_idx]
            head_k = k_split[head_idx]
            head_v = v_split[head_idx].detach().cpu().numpy()  # shape: (seq_len, head_dim)

            # Attention スコア (shape: (seq_len, seq_len))
            attn_scores = (head_q @ head_k.transpose(-2, -1)) * (1.0 / np.sqrt(head_q.size(-1)))
            attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1).detach().cpu().numpy()

            cls_attn = attn_scores[0, :]  # CLS行 (CLS→各トークンのスコア)

            # UMAP/PCA で 2 次元へ射影
            reduced = reducers_per_head[head_idx].transform(head_v)  # shape: (seq_len, 2)

            data_dict[(layer_idx, head_idx)] = {
                "reduced": reduced,
                "cls_attn": cls_attn,
            }

            # 軸そろえ用に保存
            x_vals_per_head[head_idx].append(reduced[:, 0])
            y_vals_per_head[head_idx].append(reduced[:, 1])

    # -------------------
    # ヘッドごとに x,y の最小値・最大値を計算
    # -------------------
    x_min_per_head = {}
    x_max_per_head = {}
    y_min_per_head = {}
    y_max_per_head = {}

    for head_idx in range(num_heads):
        all_x = np.concatenate(x_vals_per_head[head_idx])
        all_y = np.concatenate(y_vals_per_head[head_idx])

        x_min_per_head[head_idx] = all_x.min()
        x_max_per_head[head_idx] = all_x.max()
        y_min_per_head[head_idx] = all_y.min()
        y_max_per_head[head_idx] = all_y.max()

    # -------------------
    # ヘッド × レイヤーごとに 1枚ずつ図を作成・保存
    # -------------------
    for head_idx in range(num_heads):
        # headごとの出力フォルダ
        head_dir = os.path.join(output_dir, f"head{head_idx}")
        os.makedirs(head_dir, exist_ok=True)

        # 軸範囲を取得
        x_min, x_max = x_min_per_head[head_idx], x_max_per_head[head_idx]
        y_min, y_max = y_min_per_head[head_idx], y_max_per_head[head_idx]

        for layer_idx in range(num_layers):
            reduced = data_dict[(layer_idx, head_idx)]["reduced"]
            cls_attn = data_dict[(layer_idx, head_idx)]["cls_attn"]

            fig, ax = plt.subplots(figsize=(5, 4))

            # CLS とそれ以外でマーカーサイズを変化
            sizes = cls_attn[1:] * 1500
            cls_size = cls_attn[0] * 1500

            # 通常トークンを散布図
            ax.scatter(
                reduced[1:, 0], reduced[1:, 1],
                s=sizes, alpha=0.4, c="blue", label="tokens"
            )
            # CLSトークンを赤いマーカーで
            ax.scatter(
                reduced[0, 0], reduced[0, 1],
                s=cls_size, c="red", alpha=0.8, label="CLS"
            )

            ax.set_title(f"Layer {layer_idx}, Head {head_idx}")
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

            # トークン名を配置
            if seq_len <= 32:
                for i in range(seq_len):
                    ax.text(
                        reduced[i, 0],
                        reduced[i, 1],
                        tokens[i],
                        fontsize=6,
                        ha="center",
                        va="center",
                        color="black",
                    )

            # 画像を保存
            save_path = os.path.join(head_dir, f"layer{layer_idx}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"Plots saved under '{output_dir}/head*/layer*.png'.")


def plot_all_layers_with_shared_head_reducers(
    text: str,
    tokenizer,
    model: BertModelWithQKV,
    reducers_per_head,
    num_layers: int = 12,
    num_heads: int = 12,
    max_length=20,
    output_dir: str = None,
    output_filename: str = None,
    show_plot=True,
):
    """
    全レイヤーの v を 2 次元マッピングし、ヘッドごとに統一されたリダクションモデルを適用して可視化。
    さらに、同じヘッドであればレイヤーをまたいでも x/y 軸の表示範囲を固定する。
    """

    inputs = preprocess_text(text, tokenizer, max_length=max_length)
    with torch.no_grad():
        model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    seq_len = len(tokens)

    # -----------------------------------------------------------
    # 1st pass: 各レイヤー・各ヘッドの次元削減結果をまずまとめて計算し、メモリに保持
    # -----------------------------------------------------------
    # {(layer_idx, head_idx): {"reduced": 2次元座標, "cls_attn": (seq_len,)}}
    data_dict = {}

    # ヘッドごとの x, y 座標を格納するためのリスト (レイヤーをまたいで集計)
    x_vals_per_head = [[] for _ in range(num_heads)]
    y_vals_per_head = [[] for _ in range(num_heads)]

    for layer_idx in range(num_layers):
        q, k, v = model.get_qkv_from_layer(layer_idx)
        q_split = split_heads(q, num_heads=num_heads)[0]  # (num_heads, seq_len, head_dim)
        k_split = split_heads(k, num_heads=num_heads)[0]
        v_split = split_heads(v, num_heads=num_heads)[0]

        for head_idx in range(num_heads):
            head_q = q_split[head_idx]  # shape: (seq_len, head_dim)
            head_k = k_split[head_idx]  # shape: (seq_len, head_dim)
            head_v = v_split[head_idx].detach().cpu().numpy()  # shape: (seq_len, head_dim)

            # Attention スコアを計算 (shape: (seq_len, seq_len))
            attn_scores = (head_q @ head_k.transpose(-2, -1)) * (1.0 / np.sqrt(head_q.size(-1)))
            attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1).detach().cpu().numpy()

            # CLS トークンのスコアを取得 (shape: (seq_len,))
            cls_attn = attn_scores[0, :]  # CLS トークンのスコア (行方向)

            # UMAP/PCA で 2 次元へ射影
            reduced = reducers_per_head[head_idx].transform(head_v)

            # データを一時保存
            data_dict[(layer_idx, head_idx)] = {
                "reduced": reduced,
                "cls_attn": cls_attn,
            }

            # 軸固定のため、ヘッドごとに x, y の値を全部集めておく
            x_vals_per_head[head_idx].append(reduced[:, 0])
            y_vals_per_head[head_idx].append(reduced[:, 1])

    # -----------------------------------------------------------
    # ヘッドごとに、全レイヤー分の x, y をまとめたときの最小値 / 最大値を求める
    # -----------------------------------------------------------
    x_min_per_head = {}
    x_max_per_head = {}
    y_min_per_head = {}
    y_max_per_head = {}

    for head_idx in range(num_heads):
        all_x = np.concatenate(x_vals_per_head[head_idx])
        all_y = np.concatenate(y_vals_per_head[head_idx])
        x_min_per_head[head_idx] = all_x.min()
        x_max_per_head[head_idx] = all_x.max()
        y_min_per_head[head_idx] = all_y.min()
        y_max_per_head[head_idx] = all_y.max()

    # -----------------------------------------------------------
    # 2nd pass: 実際にプロットし、ヘッドごとに同じ x/y 範囲を設定
    # -----------------------------------------------------------
    fig, axes = plt.subplots(
        num_layers, num_heads, figsize=(4 * num_heads, 4 * num_layers), squeeze=False
    )

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            reduced = data_dict[(layer_idx, head_idx)]["reduced"]
            cls_attn = data_dict[(layer_idx, head_idx)]["cls_attn"]

            # Attention スコアに基づくサイズ設定
            sizes = cls_attn[1:] * 1500  # CLS トークン以外のトークン
            cls_tkn_size = cls_attn[0] * 1500  # CLS トークン自身

            ax = axes[layer_idx, head_idx]
            ax.scatter(reduced[1:, 0], reduced[1:, 1], s=sizes, alpha=0.5, c="blue")
            ax.scatter(
                reduced[0, 0], reduced[0, 1],
                s=cls_tkn_size, c="red", alpha=0.8, label="CLS"
            )
            ax.set_title(f"Layer {layer_idx}, Head {head_idx}")

            # 各ヘッドについて、全レイヤーで共有の x/y 軸範囲を固定
            ax.set_xlim([x_min_per_head[head_idx], x_max_per_head[head_idx]])
            ax.set_ylim([y_min_per_head[head_idx], y_max_per_head[head_idx]])

            # 目盛りを有効化
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

            # トークンを点の近くに表示
            if seq_len <= 32:
                for i in range(seq_len):
                    ax.text(
                        reduced[i, 0],
                        reduced[i, 1],
                        tokens[i],
                        fontsize=6,
                        ha="center",
                        va="center",
                        color="black",
                    )

    plt.tight_layout()

    # 保存処理
    if output_dir is not None and output_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, output_filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved at: {save_path}")

    if show_plot:
        plt.show()

    return fig


###############################################################################
# 5. 動作テスト用のメイン
###############################################################################
def main():
    # =======================
    # モデル・トークナイザ等の準備
    # =======================
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = load_bert_with_qkv(model_name)
    model.eval()

    texts_for_fitting = [
        "She was a teacher for forty years and her writing has appeared in journals and anthologies since the early 1980s."
    ]
    # texts_for_fitting = [
    #     "it’s a good thing most animated sci-fi movies come from japan, because “titan a.e.” is proof that hollywood doesn’t have a clue how to do it. i don’t know what this film is supposed to be about. from what i can tell it’s about a young man named kale who’s one of the last survivors of earth in the early 31st century who unknowingly possesses the key to saving and re-generating what is left of the human race. that’s a fine premise for an action-packed sci-fi animated movie, but there’s no payoff. the story takes the main characters all over the galaxy in their search for a legendary ship that the evil “dredge” aliens want to destroy for no apparent reason. so in the process we get a lot of spaceship fights, fistfights, blaster fights and more double-crosses than you can shake a stick at. there’s so much pointless sci-fi banter it’s too much to take. the galaxy here is a total rip-off of the “star wars” universe the creators don’t bother filling in the basic details which makes the story confusing, the characters unmotivated and superficial and the plot just plain boring. despite the fantastic animation and special effects, it’s just not an interesting movie."
    # ]

    num_layers = 12
    num_heads = 12
    use_umap = True

    # =======================
    # ヘッドごとに次元削減モデルをフィッティング
    # =======================
    reducers_per_head = fit_umap_or_pca_per_head(
        texts=texts_for_fitting,
        tokenizer=tokenizer,
        model=model,
        num_layers=num_layers,
        num_heads=num_heads,
        use_umap=use_umap,
    )

    # =======================
    # 可視化と保存
    # =======================
    text_to_plot = "She was a teacher for forty years and her writing has appeared in journals and anthologies since the early 1980s."
    # text_to_plot = "it’s a good thing most animated sci-fi movies come from japan, because “titan a.e.” is proof that hollywood doesn’t have a clue how to do it. i don’t know what this film is supposed to be about. from what i can tell it’s about a young man named kale who’s one of the last survivors of earth in the early 31st century who unknowingly possesses the key to saving and re-generating what is left of the human race. that’s a fine premise for an action-packed sci-fi animated movie, but there’s no payoff. the story takes the main characters all over the galaxy in their search for a legendary ship that the evil “dredge” aliens want to destroy for no apparent reason. so in the process we get a lot of spaceship fights, fistfights, blaster fights and more double-crosses than you can shake a stick at. there’s so much pointless sci-fi banter it’s too much to take. the galaxy here is a total rip-off of the “star wars” universe the creators don’t bother filling in the basic details which makes the story confusing, the characters unmotivated and superficial and the plot just plain boring. despite the fantastic animation and special effects, it’s just not an interesting movie."

    plot_head_layer_individual_figs(
        text=text_to_plot,
        tokenizer=tokenizer,
        model=model,
        reducers_per_head=reducers_per_head,
        num_layers=num_layers,
        num_heads=num_heads,
        max_length=512,
        output_dir="qkv_outputs",
    )

    plot_all_layers_with_shared_head_reducers(
        text=text_to_plot,
        tokenizer=tokenizer,
        model=model,
        reducers_per_head=reducers_per_head,
        num_layers=num_layers,
        num_heads=num_heads,
        max_length=512,
        output_dir="qkv_outputs",
        output_filename="all_layers_heads_values.png",
        show_plot=False,
    )

if __name__ == "__main__":
    main()
