from collections import defaultdict
from pytorch_lightning.callbacks import Callback
from sklearn.manifold import TSNE
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import itertools

import torch
import torchvision
import os


class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.count = 0

    def on_train_epoch_start(self, trainer, pl_module: pl.LightningModule):
        self.count += 1
        weight = pl_module.model.model.vit.embeddings.patch_embeddings.projection.weight
        k_size, c, h, w = weight.shape
        folder_path = f"/mnt/HDD10TB-1/sugimoto/2024_sugimoto_edge/weight_kernel_{self.count}_epoch"
        os.makedirs(folder_path, exist_ok=True)
        for k in range(k_size):
            weight_per_kernel = weight[k, :, :, :]
            weight_per_image = torchvision.transforms.functional.to_pil_image(weight_per_kernel)
            image_path = os.path.join(folder_path, f"weight_kernel_{k}.png")
            weight_per_image.save(image_path)


class MyCosineSimilarityCallback(Callback):
    def __init__(self):
        super().__init__()
        self.layer_cos_sim_list = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pl_module.model.model.vit.encoder.all_edge_features = []
        pl_module.model.model.vit.encoder.all_rgb_features = []
        pl_module.val_epoch_encoder_outputs = {"Edge": {}, "RGB": {}}
        self.layer_cos_sim_list = {i: [] for i in range(12)}

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        pl_module.model.model.vit.encoder.all_edge_features = []
        pl_module.model.model.vit.encoder.all_rgb_features = []
        pl_module.val_epoch_encoder_outputs = {"Edge": {}, "RGB": {}}

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.val_epoch_encoder_outputs = {"Edge": {}, "RGB": {}}
        self.layer_cos_sim_list = {i: [] for i in range(12)}

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # ここに各GPUの処理結果が格納される
        local_outputs = pl_module.val_epoch_encoder_outputs
        all_outputs = [None for _ in range(torch.distributed.get_world_size(group=None))]
        torch.distributed.all_gather_object(all_outputs, local_outputs)

        # 集めた結果をひとつの辞書にマージ
        merged_outputs = {"Edge": {}, "RGB": {}}
        for outputs in all_outputs:
            for key, value in outputs["Edge"].items():
                merged_outputs["Edge"][key] = value
            for key, value in outputs["RGB"].items():
                merged_outputs["RGB"][key] = value

        # 最後のcos類似度計算はメインプロセスで行う
        if trainer.global_rank == 0:
            file_name_list = list(
                set(merged_outputs["Edge"].keys())
                & set(merged_outputs["RGB"].keys())
            )

            edge_outputs = []
            rgb_outputs = []
            for file_name in file_name_list:
                edge_outputs.append(merged_outputs["Edge"][file_name])
                rgb_outputs.append(merged_outputs["RGB"][file_name])

            # (ファイル数, 層数, トークン数,出力次元)
            edge_outputs = torch.stack([torch.stack([e.to('cpu') for e in edge])
                                       for edge in edge_outputs])  # (N, 12, 197, 768)

            rgb_outputs = torch.stack([torch.stack([r.to('cpu') for r in rgb])
                                       for rgb in rgb_outputs])  # (N, 12, 197, 768)

            # 各層ごとのコサイン類似度を計算
            cos_sim_matrix = torch.cosine_similarity(
                edge_outputs, rgb_outputs, dim=-1
            )  # (N, 12, 197)
            cos_sim_matrix = cos_sim_matrix.to(dtype=torch.float32).cpu().numpy()

            # 各層ごとの結果をリストに追加
            for i in range(12):
                self.layer_cos_sim_list[i].extend(np.array(cos_sim_matrix[:, i, :]).flatten().tolist())

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # 各層ごとに類似度のヒストグラムをプロットして保存
            for layer_idx, cos_sim_list in self.layer_cos_sim_list.items():
                fig_hist = self.plot_cosine_similarity_histogram(
                    cos_sim_list, f"Edge", f"RGB"
                )
                pl_module.logger.experiment.log_figure(
                    figure_name=f"cosine_similarity_histogram_Layer_{layer_idx}_Epoch_{trainer.current_epoch}",
                    figure=fig_hist,
                    step=trainer.current_epoch,
                )

    def plot_cosine_similarity_histogram(
        self, cos_sim_list, dataset1_name, dataset2_name
    ):
        # ビンの設定 (-1 から 1 まで 0.05 間隔)
        bins = list(np.arange(-1, 1.05, 0.05))  # 階級幅0.05, -1~1まで

        # ヒストグラムのプロット
        fig, ax = plt.subplots(figsize=(20, 16))
        ax.hist(cos_sim_list, bins=bins, edgecolor="black", color="skyblue")

        # グラフの設定
        ax.set_title(
            f"Cosine Similarity Histogram between {dataset1_name} and {dataset2_name}"
        )
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Frequency")
        ax.set_xticks(np.arange(-1, 1.1, 0.2))  # -1から1まで0.2刻み
        ax.set_yticks(np.arange(0, 200001, 20000))  # 0から20000まで100刻み
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        return fig


class MytSNECallBack(Callback):
    def __init__(self, perplexity=30, max_iter=1000):
        """
        t-SNEを用いたRGBとEdge特徴量の可視化を行うCallback。

        Parameters:
        - perplexity (int): t-SNEのperplexityパラメータ。
        - max_iter (int): t-SNEの最大反復回数。
        """
        self.perplexity = perplexity
        self.max_iter = max_iter

    def on_validation_epoch_start(self, trainer, pl_module):
        # 検証用の特徴量を初期化
        pl_module.val_epoch_encoder_outputs = {"Edge": {}, "RGB": {}}

    def on_validation_epoch_end(self, trainer, pl_module):
        # EdgeとRGBで共通するファイル名を取得
        file_name_list = list(
            set(pl_module.val_epoch_encoder_outputs["Edge"].keys())
            & set(pl_module.val_epoch_encoder_outputs["RGB"].keys())
        )

        if not file_name_list:
            print("No common files between Edge and RGB features.")
            return

        # t-SNE用のデータ準備
        features = []
        labels = []
        categories = []

        for file_name in file_name_list:
            # EdgeとRGB特徴量を取得
            edge_feature = pl_module.val_epoch_encoder_outputs["Edge"][
                file_name
            ].cpu().float()
            rgb_feature = pl_module.val_epoch_encoder_outputs["RGB"][
                file_name
            ].cpu().float()

            features.append(edge_feature.numpy())
            labels.append("Edge")
            categories.append(file_name.split("_")[1])  # カテゴリを取得
            features.append(rgb_feature.numpy())
            labels.append("RGB")
            categories.append(file_name.split("_")[1])  # カテゴリを取得

        features = torch.stack([torch.tensor(f) for f in features], dim=0).numpy()

        tsne = TSNE(n_components=2, perplexity=self.perplexity, max_iter=self.max_iter, random_state=42)
        reduced_features = tsne.fit_transform(features)

        self.plot_and_log_tsne_scatter_by_category(
            trainer, pl_module, reduced_features, labels, categories
        )

    def plot_and_log_tsne_scatter_by_category(self, trainer, pl_module, reduced_features, labels, categories):
        """
        カテゴリごとにt-SNEの結果を散布図でプロットし、Cometにアップロード。

        Parameters:
        - trainer: Lightning Trainerオブジェクト。
        - pl_module: LightningModule。
        - reduced_features (numpy.ndarray): t-SNEによる次元削減結果。
        - labels (list): 各データ点のラベル（カテゴリ）。
        - categories (list): 各データ点のファイルカテゴリ。
        """
        category_to_indices = defaultdict(list)

        for i, category in enumerate(categories):
            category_to_indices[category].append(i)

        for category, indices in category_to_indices.items():
            # 新しいFigureを作成
            fig, ax = plt.subplots(figsize=(10, 8))
            unique_labels = set([labels[i] for i in indices])

            for label in unique_labels:
                label_indices = [i for i in indices if labels[i] == label]
                ax.scatter(
                    reduced_features[label_indices, 0],
                    reduced_features[label_indices, 1],
                    label=label,
                    alpha=0.7,
                )

            ax.set_title(f"t-SNE Scatter Plot - {category}")
            ax.set_xlabel("t-SNE Dimension 1")
            ax.set_ylabel("t-SNE Dimension 2")
            ax.legend()
            ax.grid(True)

            # Cometにアップロード
            pl_module.logger.experiment.log_figure(
                figure_name=f"t-SNE Scatter Plot - {category}",
                figure=fig,
                step=trainer.current_epoch,
            )

            # Figureを閉じる
            plt.close(fig)


class MyKLDivergenceCallback(Callback):
    def __init__(self):
        """
        EdgeのclstokenとRGBのclstokenのKLダイバージェンスを測定し、
        結果をヒストグラムとしてプロットするCallback。
        """
        pass

    def on_validation_epoch_start(self, trainer, pl_module):
        # 検証エポックごとのエンコーダ出力を初期化
        pl_module.val_epoch_encoder_outputs = {"Edge": {}, "RGB": {}}

    def on_validation_epoch_end(self, trainer, pl_module):
        # EdgeとRGBで共通するファイル名を取得
        file_name_list = list(
            set(pl_module.val_epoch_encoder_outputs["Edge"].keys())
            & set(pl_module.val_epoch_encoder_outputs["RGB"].keys())
        )

        if not file_name_list:
            print("No common files between Edge and RGB features.")
            return

        # KLダイバージェンス計算
        kl_div_list = []

        for file_name in file_name_list:
            edge_output = pl_module.val_epoch_encoder_outputs["Edge"][file_name].cpu().float()
            rgb_output = pl_module.val_epoch_encoder_outputs["RGB"][file_name].cpu().float()

            # ソフトマックスを適用して確率分布に変換
            edge_probs = torch.softmax(edge_output, dim=-1)
            rgb_probs = torch.softmax(rgb_output, dim=-1)

            # KLダイバージェンス計算
            kl_div = torch.nn.functional.kl_div(
                edge_probs.log(), rgb_probs, reduction="batchmean"
            ).item()

            kl_div_list.append(kl_div)

        # KLダイバージェンスをヒストグラムとしてプロット
        fig_hist = self.plot_kl_divergence_histogram(kl_div_list, "Edge", "RGB")

        # 結果をCometに保存
        pl_module.logger.experiment.log_figure(
            figure_name="KL Divergence Histogram - Edge & RGB",
            figure=fig_hist,
            step=trainer.current_epoch,
        )

        # Figureを閉じる
        plt.close(fig_hist)

    def plot_kl_divergence_histogram(self, kl_div_list, dataset1_name, dataset2_name):
        """
        KLダイバージェンスのヒストグラムをプロットする。

        Parameters:
        - kl_div_list: KLダイバージェンスのリスト。
        - dataset1_name: データセット1の名前。
        - dataset2_name: データセット2の名前。
        """
        # ビンの設定
        bins = np.linspace(0, max(kl_div_list) if kl_div_list else 1, 50)  # 0から最大値まで50分割

        # ヒストグラムのプロット
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.hist(kl_div_list, bins=bins, edgecolor="black", color="skyblue")

        # グラフの設定
        ax.set_title(f"KL Divergence Histogram between {dataset1_name} and {dataset2_name}")
        ax.set_xlabel("KL Divergence")
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        return fig
