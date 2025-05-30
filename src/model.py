# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.text import BLEUScore, CharErrorRate
import math
from typing import Dict, List, Optional
from torchvision.models import resnet18, resnet50
from timm.models.vision_transformer import VisionTransformer

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer decoder."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class Encoder(nn.Module):
    """Image encoder with configurable backbone."""
    def __init__(self, encoder_name: str = "vit_tiny", hidden_dim: int = 256):
        super().__init__()
        self.encoder_name = encoder_name
        self.feature_dim = hidden_dim
        
        if encoder_name.startswith("vit"):
            # Vision Transformer
            if encoder_name == "vit_tiny":
                self.backbone = VisionTransformer(
                    img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, 
                    num_classes=0, global_pool=''
                )
                self.proj = nn.Linear(192, hidden_dim)
            elif encoder_name == "vit_small":
                self.backbone = VisionTransformer(
                    img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6,
                    num_classes=0, global_pool=''
                )
                self.proj = nn.Linear(384, hidden_dim)
            elif encoder_name == "vit_base":
                self.backbone = VisionTransformer(
                    img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                    num_classes=0, global_pool=''
                )
                self.proj = nn.Linear(768, hidden_dim)
        else:
            # ResNet
            if encoder_name == "resnet18":
                self.backbone = resnet18(pretrained=True)
                self.backbone.fc = nn.Identity()
                self.proj = nn.Linear(512, hidden_dim)
            elif encoder_name == "resnet50":
                self.backbone = resnet50(pretrained=True)
                self.backbone.fc = nn.Identity()
                self.proj = nn.Linear(2048, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_name.startswith("vit"):
            # ViT returns (batch_size, num_patches + 1, embed_dim)
            features = self.backbone(x)
            # Use cls token or average patches
            features = features.mean(dim=1)  # (batch_size, embed_dim)
        else:
            # ResNet returns (batch_size, feature_dim)
            features = self.backbone(x)
        
        # Project to hidden_dim
        features = self.proj(features).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        return features

class LatexOCRModel(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        char2idx: Dict[str, int],
        idx2char: Dict[int, str],
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        encoder_name: str = "vit_tiny",
        num_decoder_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        max_seq_len: int = 512,
        pad_token_id: int = 0,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Vocabulary
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        
        # Metrics
        self.bleu = BLEUScore()
        self.cer = CharErrorRate()
        
        # Encoder
        self.encoder = Encoder(encoder_name, hidden_dim)
        
        # Decoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, 
        images: torch.Tensor, 
        tgt_tokens: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> torch.Tensor:
        # Encode images
        memory = self.encoder(images)  # (batch_size, 1, hidden_dim)
        
        if tgt_tokens is None:
            # Inference mode
            return self.generate(memory)
        
        # Training mode
        tgt_emb = self.embedding(tgt_tokens) * math.sqrt(self.hparams.embedding_dim)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Create masks
        tgt_mask = self._generate_square_subsequent_mask(tgt_emb.size(1)).to(images.device)
        tgt_pad_mask = (tgt_tokens == self.pad_token_id)
        
        # Decode
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        
        logits = self.output_proj(output)
        return logits

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for transformer decoder."""
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask

    def generate(self, memory: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        """Generate tokens autoregressively."""
        max_len = max_len or self.hparams.max_seq_len
        batch_size = memory.size(0)
        device = memory.device
        
        # Start with SOS token
        ys = torch.ones(batch_size, 1).fill_(self.sos_token_id).long().to(device)
        
        for _ in range(max_len - 1):
            tgt_emb = self.embedding(ys) * math.sqrt(self.hparams.embedding_dim)
            tgt_emb = self.pos_encoder(tgt_emb)
            
            tgt_mask = self._generate_square_subsequent_mask(ys.size(1)).to(device)
            output = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask
            )
            
            logits = self.output_proj(output[:, -1:])
            next_token = logits.argmax(-1)
            ys = torch.cat([ys, next_token], dim=1)
            
            # Stop if all sequences generated EOS
            if (next_token == self.eos_token_id).all():
                break
                
        return ys

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        images = batch['image']
        tgt_tokens = batch['formula']
        
        logits = self(images, tgt_tokens[:, :-1])
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt_tokens[:, 1:].reshape(-1),
            ignore_index=self.pad_token_id
        )
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        images = batch['image']
        tgt_tokens = batch['formula']
        
        # Compute loss
        logits = self(images, tgt_tokens[:, :-1])
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt_tokens[:, 1:].reshape(-1),
            ignore_index=self.pad_token_id
        )
        
        # Generate predictions
        pred_tokens = self.generate(self.encoder(images))
        pred_texts = self._tokens_to_text(pred_tokens)
        target_texts = self._tokens_to_text(tgt_tokens)
        
        # Compute metrics
        bleu = self.bleu(pred_texts, [[t] for t in target_texts])
        cer = self.cer(pred_texts, target_texts)
        
        # Log examples
        if batch_idx == 0:
            self._log_examples(pred_texts, target_texts)
        
        self.log_dict({
            'val_loss': loss,
            'val_bleu': bleu,
            'val_cer': cer
        }, prog_bar=True)
        
        return {'loss': loss, 'bleu': bleu, 'cer': cer}

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

    def _tokens_to_text(self, tokens: torch.Tensor) -> List[str]:
        """Convert token IDs to text strings."""
        texts = []
        for seq in tokens:
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            text = []
            for idx in seq:
                if idx == self.eos_token_id:
                    break
                if idx > 2:  # Skip special tokens
                    text.append(self.idx2char.get(idx, ''))
            texts.append(''.join(text))
        return texts

    def _log_examples(self, pred_texts: List[str], target_texts: List[str]):
        """Log example predictions to TensorBoard."""
        for i in range(min(3, len(pred_texts))):
            self.logger.experiment.add_text(
                f"example_{i}",
                f"Pred: {pred_texts[i]}\nTarget: {target_texts[i]}",
                self.global_step)