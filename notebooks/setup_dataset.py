import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import requests
from pathlib import Path

# Constants
DATASET_URL = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"
DATASET_PATH = "../data/UrbanSound8K.tar.gz"
EXTRACT_PATH = "../data"
METADATA_PATH = "../data/UrbanSound8K/metadata/UrbanSound8K.csv"

def download_dataset():
    """Download the UrbanSound8K dataset if it doesn't exist."""
    # if not os.path.exists(DATASET_PATH):
    print("Downloading UrbanSound8K dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    
    # Download with progress bar
    response = requests.get(DATASET_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(DATASET_PATH, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    
    print("Download complete!")
    # else:
    #     print("Dataset already downloaded.")

def extract_dataset():
    """Extract the downloaded dataset."""
    extracted_folder = os.path.join(EXTRACT_PATH, "UrbanSound8K")
    # if not os.path.exists(extracted_folder):
    print("Extracting dataset...")
    try:
        with tarfile.open(DATASET_PATH, "r:gz") as tar:
            # Get total number of files for progress bar
            total_files = len(tar.getmembers())
            
            # Extract with progress bar
            for member in tqdm(tar.getmembers(), total=total_files, desc="Extracting"):
                tar.extract(member, path=EXTRACT_PATH)
        print("Extraction complete!")
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        # Clean up partial extraction
        if os.path.exists(extracted_folder):
            import shutil
            shutil.rmtree(extracted_folder)
        raise
    # else:
    #     print("Dataset already extracted.")

def load_metadata():
    """Load and return the dataset metadata."""
    return pd.read_csv(METADATA_PATH)

def explore_dataset(metadata):
    """Explore the dataset and display basic statistics."""
    print("\nDataset Overview:")
    print("-" * 50)
    print(f"Total number of audio files: {len(metadata)}")
    
    # Display class distribution
    class_dist = metadata['class'].value_counts()
    print("\nClass Distribution:")
    print("-" * 50)
    for class_name, count in class_dist.items():
        print(f"{class_name}: {count} files")
    
    # Create a bar plot of class distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_dist.index, y=class_dist.values)
    plt.title("Distribution of Sound Classes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../data/class_distribution.png")
    plt.close()

def visualize_spectrograms(metadata, num_samples=3):
    """Visualize spectrograms for a few samples from each class."""
    classes = metadata['class'].unique()
    
    for class_name in classes:
        print(f"\nVisualizing spectrograms for class: {class_name}")
        class_samples = metadata[metadata['class'] == class_name].sample(n=num_samples)
        
        for _, sample in class_samples.iterrows():
            # Construct the full path to the audio file
            audio_path = os.path.join(EXTRACT_PATH, "UrbanSound8K", "audio", 
                                    f"fold{sample['fold']}", sample['slice_file_name'])
            
            # Load and process the audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Create mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Plot
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel Spectrogram - {class_name}\n{sample["slice_file_name"]}')
            plt.tight_layout()
            
            # Save the plot
            save_path = os.path.join(EXTRACT_PATH, f"spectrogram_{class_name}_{sample['slice_file_name'].replace('.wav', '.png')}")
            plt.savefig(save_path)
            plt.close()

def main():
    # Create data directory if it doesn't exist
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    
    # Download and extract dataset
    download_dataset()
    extract_dataset()
    
    # Load and explore metadata
    metadata = load_metadata()
    explore_dataset(metadata)
    
    # Visualize spectrograms
    print("\nGenerating spectrograms...")
    visualize_spectrograms(metadata)
    print("\nExploration complete! Check the data directory for visualizations.")

if __name__ == "__main__":
    main() 