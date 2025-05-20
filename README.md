# NoiseSense: UrbanSound8K Audio Classification

## Overview
NoiseSense is a machine learning pipeline for classifying environmental sounds using the UrbanSound8K dataset. The project includes scripts for dataset setup, preprocessing, and model training.

## Project Structure
```
NoiseSense/
├── data/
│   ├── raw/              # Raw UrbanSound8K data (after extraction)
│   ├── processed/        # Processed data (X.npy, y.npy)
│   └── visualizations/   # Plots and figures
├── notebooks/
│   ├── setup_dataset.py      # Download and extract UrbanSound8K
│   ├── process_dataset.py    # Preprocess audio to spectrograms (X.npy, y.npy)
│   └── train_model.py        # Train the CNN model
├── requirements.txt
└── README.md
```

## Setup
1. **Clone the repository:**
   ```bash
git clone https://github.com/DiamondKJ/NoiseSense.git
cd NoiseSense
```
2. **Install dependencies:**
   ```bash
pip install -r requirements.txt
```

## Usage
1. **Download and extract the dataset:**
   ```bash
python notebooks/setup_dataset.py
```
2. **Preprocess the data:**
   ```bash
python notebooks/process_dataset.py
```
   This will generate `data/processed/X.npy` and `data/processed/y.npy`.
3. **Train the model:**
   ```bash
python notebooks/train_model.py
```
   This will save model weights and training visualizations in the appropriate folders.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## Notes
- The `data/` directory is ignored by git (see `.gitignore`). You will need to run the setup and preprocessing scripts to generate the necessary files.
- For best results, run scripts in order: setup → preprocess → train.

## License
MIT License

Copyright (c) 2024 Kaustutubh Joshi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

echo "data/\n__pycache__/\n*.pyc\n*.pyo\n*.pyd\n*.env\n.venv\nenv/\nvenv/\nENV/\n.ipynb_checkpoints/" > .gitignore 
