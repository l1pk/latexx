import pytorch_lightning as pl
import torch.nn as nn

class FormulaModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 126 * 126, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)  # output_size is the number of unique LaTeX tokens
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        loss = nn.CrossEntropyLoss()(output, labels)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
