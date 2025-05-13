import pytorch_lightning as pl
from data_module import FormulaDataModule
from model import FormulaModel
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config.yaml")
def main(config: DictConfig):
    data_module = FormulaDataModule(config.data.img_folder, config.data.annotations)
    model = FormulaModel()

    trainer = pl.Trainer(max_epochs=config.training.epochs)
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
