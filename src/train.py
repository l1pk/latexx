# src/train.py
import os
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from model import LatexOCRModel
from data_module import LatexOCRDataModule

@hydra.main(config_path="../", config_name="config", version_base="1.3")
def train(config: DictConfig):
    # Set random seeds for reproducibility
    pl.seed_everything(config.training.seed)
    
    # Initialize data module
    data_module = LatexOCRDataModule(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        train_val_test_split=config.data.train_val_test_split,
        img_size=config.data.img_size,
        max_seq_len=config.data.max_seq_len
    )
    
    # Build model
    model = LatexOCRModel(
        vocab_size=len(data_module.idx2char),
        char2idx=data_module.char2idx,
        idx2char=data_module.idx2char,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        encoder_name=config.model.encoder_name,
        num_decoder_layers=config.model.num_decoder_layers,
        nhead=config.model.nhead,
        dropout=config.model.dropout,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_seq_len=config.data.max_seq_len,
        pad_token_id=config.model.pad_token_id,
        sos_token_id=config.model.sos_token_id,
        eos_token_id=config.model.eos_token_id
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_bleu",
        mode="max",
        filename="best-{epoch}-{val_bleu:.2f}",
        save_top_k=1,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor="val_bleu",
        patience=config.training.patience,
        mode="max"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = RichProgressBar()
    
    # Loggers
    loggers = [
        TensorBoardLogger(
            save_dir=config.logging.log_dir,
            name=config.logging.experiment_name,
            version=os.environ.get("SLURM_JOB_ID", None)
        )
    ]
    
    if config.logging.use_wandb:
        loggers.append(
            WandbLogger(
                project=config.logging.wandb_project,
                name=config.logging.experiment_name,
                log_model="all"
            )
        )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stopping, lr_monitor, progress_bar],
        logger=loggers,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        log_every_n_steps=config.logging.log_every_n_steps,
        deterministic=True
    )
    
    # Train
    trainer.fit(model, datamodule=data_module)
    
    # Test
    trainer.test(model, datamodule=data_module, ckpt_path="best")

if __name__ == "__main__":
    train()