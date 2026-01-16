import argparse
import os
import torch


import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint


from utils import compute_topk_accuracy
from model import configure_model, ModelConfig
from setup import configure_optimizer, configure_scheduler

from dataset.wds_folder.actionswap_type import ActionSwapType


class SimpleLightningModel(pl.LightningModule):
    """A simple lightning module

        see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#methods
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#properties
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks

        https://lightning.ai/docs/pytorch/stable/starter/style_guide.html#method-order
    """

    def __init__(
            self,
            command_line_args: argparse.Namespace,
            n_classes: int,
            exp_name: str,
    ):
        """constructor

        see
        https://lightning.ai/docs/pytorch/stable/starter/style_guide.html#init

        Args:
            command_line_args (argparse): args
            n_classes (int): number of categories
            exp_name (str): experiment name of comet.ml
        """
        super().__init__()
        self.args = command_line_args
        self.exp_name = exp_name

        moexda_dict = {
            "moment_type": self.args.norm_type,  # "pono" or "in"
            "exchange_position": self.args.position_moex,  # e.g. "AfterMHA"
            "exchange_direction": self.args.exchange_direction,  # e.g. "edge_to_rgb"
            "stop_gradient": self.args.stop_gradient,  # bool
        }
        self.model = configure_model(ModelConfig(
            model_name=self.args.model_name,
            use_pretrained=self.args.use_pretrained,
            torch_home=self.args.torch_home,
            n_classes=n_classes,
            checkpoint_path=self.args.checkpoint_path,
            moex_layers=self.args.moex_layers,
            cos_sim=self.args.cos_sim,
            moexda_dict=moexda_dict,
        ))

        self.val_epoch_encoder_outputs = {}

        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#save-hyperparameters
        self.save_hyperparameters()

    def configure_optimizers(self):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """

        optimizer = configure_optimizer(
            optimizer_name=self.args.optimizer_name,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            momentum=self.args.momentum,
            model_params=self.model.parameters()
        )
        scheduler = configure_scheduler(
            optimizer=optimizer,
            scheduler=self.args.scheduler
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_callbacks(self):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-callbacks
        """

        save_checkpoint_dir = os.path.join(self.args.save_checkpoint_dir, self.exp_name)
        if self.global_rank == 0:
            os.makedirs(save_checkpoint_dir, exist_ok=True)

        checkpoint_callbacks = [
            ModelCheckpoint(
                dirpath=save_checkpoint_dir,
                monitor="edge_val_top1",
                mode="max",  # larger is better
                save_top_k=2,
                filename="epoch{epoch}_step{step}_acc={rgb_val_top1:.2f}",
                auto_insert_metric_name=False,
            ),
        ]

        return checkpoint_callbacks

    def log_train_loss_top15(self, loss, top1, top5, batch_size):
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-epoch-level-metrics
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log_dict
        self.log_dict(
            {
                "train_loss": loss.item(),
                "train_top1": top1,
            },
            prog_bar=True,  # show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log
        self.log(
            "train_top5",
            top5,
            prog_bar=False,  # do not show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

    def log_only_train_loss(self, loss, batch_size):
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-epoch-level-metrics
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log_dict
        self.log_dict(
            {
                "total_train_loss": loss.item(),
            },
            prog_bar=True,  # show on the progress bar
            on_step=False,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

    def edge_log_train_loss_top15(self, edge_loss, edge_top1, edge_top5, batch_size):
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-epoch-level-metrics
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log_dict
        self.log_dict(
            {
                "edge_train_loss": edge_loss.item(),
                "edge_train_top1": edge_top1,
            },
            prog_bar=True,  # show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log
        self.log(
            "edge_train_top5",
            edge_top5,
            prog_bar=False,  # do not show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

    def rgb_log_train_loss_top15(self, rgb_loss, rgb_top1, rgb_top5, batch_size):
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-epoch-level-metrics
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log_dict
        self.log_dict(
            {
                "rgb_train_loss": rgb_loss.item(),
                "rgb_train_top1": rgb_top1,
            },
            prog_bar=True,  # show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log
        self.log(
            "rgb_train_top5",
            rgb_top5,
            prog_bar=False,  # do not show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

    def training_step(self, batch, batch_idx):
        """a single training step for a batch

        Args:
            batch (Tuple[tensor]): a batch of data samples and labels
                (actual type depends on the dataloader)
            batch_idx (int): index of the batch in the epoch

        Returns:
            tensor: loss (used for backward by lightning)

        Note:
            DO NOT USE .to() or model.train() here
                (automatically send to multi-GPUs)
            DO NOT USE loss.backward() here
                (automatically performed by lightning)
            see
                https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-loop
                https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.training_step
        """
        if self.args.use_moex and not self.args.dataset_name == "VideoFolder":
            data, labels, *_, = batch
            sobel_data_list = [slist[0] for slist in data]
            sobel_data = torch.stack(sobel_data_list, dim=0)
            rgb_data_list = [rlist[1] for rlist in data]
            rgb_data = torch.stack(rgb_data_list, dim=0)

            batch_size = sobel_data.size(0)
            outputs = self.model(pixel_values=sobel_data, rgb_values=rgb_data, labels=labels)
            edge_loss = outputs.loss[0]
            rgb_loss = outputs.loss[1]
            loss = edge_loss + rgb_loss / 2
            edge_top1, edge_top5, *_ = compute_topk_accuracy(outputs.logits[0], labels, topk=(1, 5))
            rgb_top1, rgb_top5, *_ = compute_topk_accuracy(outputs.logits[1], labels, topk=(1, 5))
            self.log_only_train_loss(loss, batch_size)
            self.edge_log_train_loss_top15(edge_loss, edge_top1, edge_top5, batch_size)
            self.rgb_log_train_loss_top15(rgb_loss, rgb_top1, rgb_top5, batch_size)
        elif self.args.use_moex and self.args.dataset_name == "VideoFolder":
            data, labels, *_, = batch
            sobel_data = data[0]
            rgb_data = data[1]

            batch_size = sobel_data.size(0)
            outputs = self.model(pixel_values=sobel_data, rgb_values=rgb_data, labels=labels)
            edge_loss = outputs.loss[0]
            rgb_loss = outputs.loss[1]
            loss = edge_loss + rgb_loss / 2
            edge_top1, edge_top5, *_ = compute_topk_accuracy(outputs.logits[0], labels, topk=(1, 5))
            rgb_top1, rgb_top5, *_ = compute_topk_accuracy(outputs.logits[1], labels, topk=(1, 5))
            self.log_only_train_loss(loss, batch_size)
            self.edge_log_train_loss_top15(edge_loss, edge_top1, edge_top5, batch_size)
            self.rgb_log_train_loss_top15(rgb_loss, rgb_top1, rgb_top5, batch_size)
        else:
            data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}
            batch_size = data.size(0)
            outputs = self.model(data, labels=labels)
            loss = outputs.loss
            top1, top5, *_ = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))
            self.log_train_loss_top15(loss, top1, top5, batch_size)

        return loss

    def log_val_loss_top15(self, loss, top1, top5, batch_size):
        self.log_dict(
            {
                "val_loss": loss.item(),
                "val_top1": top1,
                "val_top5": top5,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,  # sync log metrics for validation
            batch_size=batch_size,
        )

    def edge_log_val_loss_top15(self, edge_loss, edge_top1, edge_top5, batch_size):
        self.log_dict(
            {
                "edge_val_loss": edge_loss.item(),
                "edge_val_top1": edge_top1,
                "edge_val_top5": edge_top5,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,  # sync log metrics for validation
            batch_size=batch_size,
        )

    def rgb_log_val_loss_top15(self, rgb_loss, rgb_top1, rgb_top5, batch_size):
        self.log_dict(
            {
                "rgb_val_loss": rgb_loss.item(),
                "rgb_val_top1": rgb_top1,
                "rgb_val_top5": rgb_top5,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,  # sync log metrics for validation
            batch_size=batch_size,
        )

    def log_only_val_loss(self, loss, batch_size):
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-epoch-level-metrics
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log_dict
        self.log_dict(
            {
                "total_val_loss": loss.item(),
            },
            prog_bar=False,  # show on the progress bar
            on_step=False,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

    def validation_step(self, batch, batch_idx):
        """a single validation step for a batch

        Args:
            batch (Tuple[tensor]): a batch of data samples and labels
                (actual type depends on the dataloader)
            batch_idx (int): index of the batch in the epoch

        Note:
            DO NOT USE .to() or model.eval() here
                (automatically send to multi-GPUs)
            DO NOT USE with torch.no_grad() here
                (automatically handled by lightning)
            see
                https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation
                https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.validation_step
        """
        if not self.args.use_moex and self.args.use_hat:
            video = batch[ActionSwapType.Original.value]
            swap_video = batch[ActionSwapType.ActionSwap.value]
            bg_video = batch[ActionSwapType.PersonInpainting.value]
            person_only_video = batch[ActionSwapType.PersonOnly.value]

            labels = torch.tensor(batch["label"]).to(video.device)
            bg_labels = torch.tensor(batch["bg_label"]).to(video.device)  # swap videoの背景 と bg_only に用いた動画のラベル

            batch_size = video.shape[0]

            outputs = self.model(video, labels=labels)
            loss = outputs.loss

            swap_video_logits = self.model(swap_video).logits
            bg_video_logits = self.model(bg_video).logits
            po_video_logits = self.model(person_only_video).logits

            top1, top5, *_ = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))
            shacc_top1, shacc_top5, *_ = compute_topk_accuracy(swap_video_logits, labels, topk=(1, 5))
            sberr_top1, *_ = compute_topk_accuracy(swap_video_logits, bg_labels, topk=(1,))
            bg_top1, bg_top5, *_ = compute_topk_accuracy(bg_video_logits, bg_labels, topk=(1, 5))
            po_top1, po_top5, *_ = compute_topk_accuracy(po_video_logits, labels, topk=(1, 5))

            # hat metrics
            # BOR Background only accuracy: Bg / Original
            bor = (bg_top1 / top1) * 100. if top1 > 0.0 else 0.0
            # HOR Human only accuracy: Person / Original
            hor = (po_top1 / top1) * 100 if top1 > 0.0 else 0.0
            # SHAcc Swap Human Accuracy: Aciton swap
            shacc = shacc_top1
            # SBErr Swap Background Error: Action swapの背景ラベルが予測になった割合
            sberr = sberr_top1

            self.log_val_loss_top15(loss, top1, top5, batch_size)

            self.log_dict(
                {
                    "BO_top1": bg_top1,
                    "HO_top1": po_top1,
                    "BOR": bor,
                    "HOR": hor,
                    "SHAcc": shacc,
                    "SBErr": sberr,
                },
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                rank_zero_only=False,
                sync_dist=True,  # sync log metrics for validation
                batch_size=batch_size,
            )
        elif self.args.use_hat and self.args.use_moex:
            video = batch[ActionSwapType.Original.value]
            swap_video_batch = batch[ActionSwapType.ActionSwap.value]
            bg_video_batch = batch[ActionSwapType.PersonInpainting.value]
            person_only_video_batch = batch[ActionSwapType.PersonOnly.value]

            sobel_data_list = [slist[0] for slist in video]
            sobel_data = torch.stack(sobel_data_list, dim=0)
            rgb_data_list = [rlist[1] for rlist in video]
            rgb_data = torch.stack(rgb_data_list, dim=0)
            labels = torch.tensor(batch["label"]).to(rgb_data.device)
            bg_labels = torch.tensor(batch["bg_label"]).to(rgb_data.device)  # swap videoの背景 と bg_only に用いた動画のラベル

            batch_size = rgb_data.shape[0]

            outputs = self.model(sobel_data, rgb_data, labels=labels)
            edge_loss = outputs.loss[0]
            rgb_loss = outputs.loss[1]
            loss = edge_loss + rgb_loss / 2

            edge_top1, edge_top5, *_ = compute_topk_accuracy(outputs.logits[0], labels, topk=(1, 5))
            rgb_top1, rgb_top5, *_ = compute_topk_accuracy(outputs.logits[1], labels, topk=(1, 5))

            edge_swap_video_list = [elist[0] for elist in swap_video_batch]
            edge_swap_video = torch.stack(edge_swap_video_list, dim=0)
            rgb_swap_video_list = [rlist[1] for rlist in swap_video_batch]
            rgb_swap_video = torch.stack(rgb_swap_video_list, dim=0)

            edge_bg_video_list = [elist[0] for elist in bg_video_batch]
            edge_bg_video = torch.stack(edge_bg_video_list, dim=0)
            rgb_bg_video_list = [rlist[1] for rlist in bg_video_batch]
            rgb_bg_video = torch.stack(rgb_bg_video_list, dim=0)

            edge_person_only_video_list = [slist[0] for slist in person_only_video_batch]
            edge_person_only_video = torch.stack(edge_person_only_video_list, dim=0)
            rgb_person_only_video_list = [rlist[1] for rlist in person_only_video_batch]
            rgb_person_only_video = torch.stack(rgb_person_only_video_list, dim=0)

            edge_swap_video_logits, rgb_swap_video_logits = self.model(edge_swap_video, rgb_swap_video).logits
            edge_bg_video_logits, rgb_bg_video_logits = self.model(edge_bg_video, rgb_bg_video).logits
            edge_po_video_logits, rgb_po_video_logits = self.model(edge_person_only_video, rgb_person_only_video).logits

            self.log_only_val_loss(loss, batch_size)
            self.edge_log_val_loss_top15(edge_loss, edge_top1, edge_top5, batch_size)
            self.rgb_log_val_loss_top15(rgb_loss, rgb_top1, rgb_top5, batch_size)

            edge_shacc_top1, _, *_ = compute_topk_accuracy(edge_swap_video_logits, labels, topk=(1, 5))
            edge_sberr_top1, *_ = compute_topk_accuracy(edge_swap_video_logits, bg_labels, topk=(1,))
            edge_bg_top1, _, *_ = compute_topk_accuracy(edge_bg_video_logits, bg_labels, topk=(1, 5))
            edge_po_top1, _, *_ = compute_topk_accuracy(edge_po_video_logits, labels, topk=(1, 5))
            rgb_shacc_top1, _, *_ = compute_topk_accuracy(rgb_swap_video_logits, labels, topk=(1, 5))
            rgb_sberr_top1, *_ = compute_topk_accuracy(rgb_swap_video_logits, bg_labels, topk=(1,))
            rgb_bg_top1, _, *_ = compute_topk_accuracy(rgb_bg_video_logits, bg_labels, topk=(1, 5))
            rgb_po_top1, _, *_ = compute_topk_accuracy(rgb_po_video_logits, labels, topk=(1, 5))

            # hat metrics
            # BOR Background only accuracy: Bg / Original
            edge_bor = (edge_bg_top1 / edge_top1) * 100. if edge_top1 > 0.0 else 0.0
            # HOR Human only accuracy: Person / Original
            edge_hor = (edge_po_top1 / edge_top1) * 100 if edge_top1 > 0.0 else 0.0
            # BOR Background only accuracy: Bg / Original
            rgb_bor = (rgb_bg_top1 / rgb_top1) * 100. if rgb_top1 > 0.0 else 0.0
            # HOR Human only accuracy: Person / Original
            rgb_hor = (rgb_po_top1 / rgb_top1) * 100 if rgb_top1 > 0.0 else 0.0
            # SHAcc Swap Human Accuracy: Aciton swap
            edge_shacc = edge_shacc_top1
            # SBErr Swap Background Error: Action swapの背景ラベルが予測になった割合
            edge_sberr = edge_sberr_top1
            # SHAcc Swap Human Accuracy: Aciton swap
            rgb_shacc = rgb_shacc_top1
            # SBErr Swap Background Error: Action swapの背景ラベルが予測になった割合
            rgb_sberr = rgb_sberr_top1

            self.log_dict(
                {
                    "edge_BO_top1": edge_bg_top1,
                    "edge_HO_top1": edge_po_top1,
                    "rgb_BO_top1": rgb_bg_top1,
                    "rgb_HO_top1": rgb_po_top1,
                    "edge_BOR": edge_bor,
                    "edge_HOR": edge_hor,
                    "RGB_BOR": rgb_bor,
                    "RGB_HOR": rgb_hor,
                    "edge_SHAcc": edge_shacc,
                    "edge_SBErr": edge_sberr,
                    "rgb_SHAcc": rgb_shacc,
                    "rgb_SBErr": rgb_sberr,
                },
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                rank_zero_only=False,
                sync_dist=True,  # sync log metrics for validation
                batch_size=batch_size,
            )
        elif self.args.use_moex and not self.args.use_hat:
            data, labels, filename = batch
            sobel_data_list = [slist[0] for slist in data]
            sobel_data = torch.stack(sobel_data_list, dim=0)
            rgb_data_list = [rlist[1] for rlist in data]
            rgb_data = torch.stack(rgb_data_list, dim=0)

            batch_size = sobel_data.size(0)
            outputs = self.model(pixel_values=sobel_data, rgb_values=rgb_data, labels=labels)
            edge_loss = outputs.loss[0]
            rgb_loss = outputs.loss[1]
            loss = edge_loss + rgb_loss / 2
            edge_top1, edge_top5, *_ = compute_topk_accuracy(outputs.logits[0], labels, topk=(1, 5))
            rgb_top1, rgb_top5, *_ = compute_topk_accuracy(outputs.logits[1], labels, topk=(1, 5))
            self.log_only_val_loss(loss, batch_size)
            self.edge_log_val_loss_top15(edge_loss, edge_top1, edge_top5, batch_size)
            self.rgb_log_val_loss_top15(rgb_loss, rgb_top1, rgb_top5, batch_size)

            if self.args.cos_sim:
                b, n, c = self.model.model.vit.encoder.all_edge_features[0].shape

                edge_features_list = self.model.model.vit.encoder.all_edge_features
                edge_features_list = [token.view(batch_size, int(b / batch_size), n, c) for token in edge_features_list]
                edge_features_list = [torch.mean(token, dim=1) for token in edge_features_list]

                rgb_features_list = self.model.model.vit.encoder.all_rgb_features
                rgb_features_list = [token.view(batch_size, int(b / batch_size), n, c) for token in rgb_features_list]
                rgb_features_list = [torch.mean(token, dim=1) for token in rgb_features_list]

                for i, fname in enumerate(filename):
                    edge_features = [token[i] for token in edge_features_list]
                    self.val_epoch_encoder_outputs["Edge"][fname] = edge_features
                    rgb_features = [token[i] for token in rgb_features_list]
                    self.val_epoch_encoder_outputs["RGB"][fname] = rgb_features
        else:
            data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}
            batch_size = data.size(0)
            outputs = self.model(data, labels=labels)
            loss = outputs.loss
            top1, top5, *_ = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))
            self.log_val_loss_top15(loss, top1, top5, batch_size)
