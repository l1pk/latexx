import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.text import BLEUScore, WordErrorRate
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class LatexOCRModel(LightningModule):
    def __init__(self, 
                 vocab_size,
                 embedding_dim=128,
                 hidden_dim=128,
                 num_decoder_layers=1,
                 nhead=4,
                 dropout=0.1,
                 learning_rate=1e-3,
                 weight_decay=1e-4,
                 max_seq_len=32,
                 pad_token_id=0,
                 sos_token_id=1,
                 eos_token_id=2,
                 char2idx=None,
                 idx2char=None,
                 encoder_name="identity"):
        super().__init__()
        self.save_hyperparameters()

        self.char2idx = char2idx
        self.idx2char = idx2char
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id

        self.bleu = BLEUScore()
        self.wer = WordErrorRate()

        # Минимальный "энкодер" (убираем ResNet для проверки пайплайна)
        self.encoder = nn.Identity()
        self.enc_projection = nn.Identity()
        self.feature_dim = hidden_dim

        # Декодер
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.embedding2hidden = nn.Linear(embedding_dim, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, imgs, tgt_tokens=None, teacher_forcing_ratio=1.0):
        # imgs не используются, т.к. encoder = Identity
        batch_size = imgs.size(0)
        memory = torch.zeros(batch_size, 1, self.feature_dim, device=imgs.device)
        tgt_inputs = tgt_tokens[:, :-1]
        tgt_embeddings = self.embedding(tgt_inputs) * math.sqrt(self.hparams.embedding_dim)
        tgt_embeddings = self.pos_encoder(tgt_embeddings)
        tgt_embeddings = self.embedding2hidden(tgt_embeddings)
        tgt_mask = self._generate_square_subsequent_mask(tgt_embeddings.size(1)).to(imgs.device)
        output = self.transformer_decoder(
            tgt_embeddings,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=None,
            tgt_key_padding_mask=(tgt_inputs == self.pad_token_id)
        )
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
        ys = torch.ones(batch_size, 1).fill_(self.sos_token_id).long().to(device)
        for i in range(max_len - 1):
            tgt_embeddings = self.embedding(ys) * math.sqrt(self.hparams.embedding_dim)
            tgt_embeddings = self.pos_encoder(tgt_embeddings)
            tgt_embeddings = self.embedding2hidden(tgt_embeddings)
            tgt_mask = self._generate_square_subsequent_mask(ys.size(1)).to(device)
            output = self.transformer_decoder(
                tgt_embeddings,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=None,
                tgt_key_padding_mask=(ys == self.pad_token_id)
            )
            output = self.output_projection(output)
            output = output[:, -1]
            _, next_token = torch.max(output, dim=-1)
            next_token = next_token.unsqueeze(1)
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == self.eos_token_id).all():
                break
        return ys

    def training_step(self, batch, batch_idx):
        imgs = batch['image']
        tgt_tokens = batch['formula']
        logits = self(imgs, tgt_tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_tokens[:, 1:].reshape(-1),
            ignore_index=self.pad_token_id
        )
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs = batch['image']
        tgt_tokens = batch['formula']
        logits = self(imgs, tgt_tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_tokens[:, 1:].reshape(-1),
            ignore_index=self.pad_token_id
        )
        # Для генерации memory = encoder(imgs) = zeros
        batch_size = imgs.size(0)
        memory = torch.zeros(batch_size, 1, self.feature_dim, device=imgs.device)
        predictions = self.generate(memory)
        pred_texts = self._tokens_to_text(predictions)
        target_texts = self._tokens_to_text(tgt_tokens)
        bleu_score = self.bleu(pred_texts, [[t] for t in target_texts])
        wer_score = self.wer(pred_texts, target_texts)

        if batch_idx == 0:
            print("Пример предсказания:", pred_texts[0])
            print("Эталон:", target_texts[0])
            print("Pred tokens:", predictions[0].tolist())
            print("Target tokens:", tgt_tokens[0].tolist())

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
        return optimizer

    def _tokens_to_text(self, token_ids):
        texts = []
        for seq in token_ids:
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            text = ""
            for idx in seq:
                if idx == self.eos_token_id:
                    break
                if idx > 2:
                    text += self.idx2char.get(idx, "")
            texts.append(text)
        return texts
