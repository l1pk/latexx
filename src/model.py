import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from pytorch_lightning import LightningModule
from torchvision import models
from torchmetrics.text import BLEUScore, WordErrorRate
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LatexOCRModel(LightningModule):
    def __init__(self, 
                 vocab_size=10000,
                 embedding_dim=256,
                 hidden_dim=512,
                 encoder_name="resnet50",
                 num_decoder_layers=6,
                 nhead=8,
                 dropout=0.1,
                 learning_rate=3e-4,
                 weight_decay=1e-4,
                 max_seq_len=512,
                 pad_token_id=0,
                 sos_token_id=1,
                 eos_token_id=2):
        super().__init__()
        self.save_hyperparameters()
        
        # Metrics
        self.bleu = BLEUScore()
        self.wer = WordErrorRate()
        
        # Vision encoder (CNN backbone)
        if encoder_name == "resnet18":
            self.encoder = models.resnet18(pretrained=True)
        elif encoder_name == "resnet34":
            self.encoder = models.resnet34(pretrained=True)
        elif encoder_name == "resnet50":
            self.encoder = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")
            
        # Remove classification head
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        # Projection from CNN features to transformer dimension
        self.feature_dim = self._get_encoder_feature_dim()
        self.enc_projection = nn.Linear(self.feature_dim, hidden_dim)
        
        # Decoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Special token IDs
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.max_seq_len = max_seq_len
        
    def _get_encoder_feature_dim(self):
        # Forward pass dummy input to get feature dimension
        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            x = self.encoder(x)
        return x.shape[1]  # Return channel dimension
    
    def forward(self, imgs, tgt_tokens=None, teacher_forcing_ratio=1.0):
        batch_size = imgs.size(0)
        
        features = self.encoder(imgs)
        h, w = features.shape[2], features.shape[3]
        features = features.permute(0, 2, 3, 1).reshape(features.size(0), h * w, -1)
        features = self.enc_projection(features)
        predictions = self.generate(features)
        
        if tgt_tokens is None:  # Inference mode
            return self.generate(features)
        
        # Prepare target for training with teacher forcing
        tgt = tgt_tokens[:, :-1]  # Remove last token
        tgt_embeddings = self.embedding(tgt) * math.sqrt(self.hparams.embedding_dim)
        tgt_embeddings = self.pos_encoder(tgt_embeddings.transpose(0, 1)).transpose(0, 1)
        
        # Create target mask to prevent attention to padding tokens
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Create padding mask
        tgt_padding_mask = (tgt == self.pad_token_id)
        
        # Decode
        output = self.transformer_decoder(
            tgt_embeddings, 
            features, 
            tgt_mask=tgt_mask,
            memory_key_padding_mask=None,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask
    
    def generate(self, memory, max_len=None):
        if max_len is None:
            max_len = self.max_seq_len
            
        batch_size = memory.size(0)
        device = memory.device
        
        # Start with SOS tokens
        ys = torch.ones(batch_size, 1).fill_(self.sos_token_id).long().to(device)
        
        for i in range(max_len - 1):
            # Embed target tokens
            tgt_embeddings = self.embedding(ys) * math.sqrt(self.hparams.embedding_dim)
            tgt_embeddings = self.pos_encoder(tgt_embeddings.transpose(0, 1)).transpose(0, 1)
            
            # Create target mask
            tgt_mask = self._generate_square_subsequent_mask(ys.size(1)).to(device)
            
            # Create padding mask
            tgt_padding_mask = (ys == self.pad_token_id)
            
            # Decode
            output = self.transformer_decoder(
                tgt_embeddings, 
                memory,
                tgt_mask=tgt_mask, 
                memory_key_padding_mask=None,
                tgt_key_padding_mask=tgt_padding_mask
            )
            
            # Project to vocabulary
            output = self.output_projection(output)
            
            # Take last position
            output = output[:, -1]
            
            # Select next token (greedy decoding)
            _, next_token = torch.max(output, dim=-1)
            next_token = next_token.unsqueeze(1)
            
            # Append to sequence
            ys = torch.cat([ys, next_token], dim=1)
            
            # Break if all sequences have EOS
            if (next_token == self.eos_token_id).all():
                break
                
        return ys
    
    def training_step(self, batch, batch_idx):
        imgs = batch['image']
        formulas = batch['formula']  # Assume tokenized in dataloader
        tgt_tokens = formulas
        
        # Forward pass
        logits = self(imgs, tgt_tokens)
        
        # Calculate loss (cross-entropy)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            tgt_tokens[:, 1:].reshape(-1),  # Shift by 1 to align with predictions
            ignore_index=self.pad_token_id
        )
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        imgs = batch['image']
        formulas = batch['formula']  # Assume tokenized
        tgt_tokens = formulas
        
        # Forward pass
        logits = self(imgs, tgt_tokens)
        
        # Calculate loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_tokens[:, 1:].reshape(-1),
            ignore_index=self.pad_token_id
        )
        
        features = self.encoder(imgs)  # [B, C, H, W]
        h, w = features.shape[2], features.shape[3]
        features = features.permute(0, 2, 3, 1).reshape(features.size(0), h * w, -1)  # [B, H*W, C]
        features = self.enc_projection(features)  # [B, H*W, hidden_dim]
        predictions = self.generate(features)
                
        # Calculate BLEU and WER
        pred_texts = self._tokens_to_text(predictions)
        target_texts = self._tokens_to_text(tgt_tokens)
        
        bleu_score = self.bleu(pred_texts, [[t] for t in target_texts])
        wer_score = self.wer(pred_texts, target_texts)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_bleu', bleu_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_wer', wer_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss, 'bleu': bleu_score, 'wer': wer_score}
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=self.hparams.learning_rate / 10
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
    
    def _tokens_to_text(self, token_ids):
        # This is a placeholder - in real implementation would use tokenizer
        # to convert token IDs back to text
        return ["placeholder_text" for _ in range(len(token_ids))]
