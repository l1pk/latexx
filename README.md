Запуск проекта

1. Установить зависимости:

pip install -r requirements.txt

2. Запустить обучение:

python src/train.py

3. Инференс на изображении:

python src/inference.py --image_path sample_image.png

Основные зависимости

- PyTorch Lightning

- Hydra

- TorchMetrics

- PIL

- opencv-python