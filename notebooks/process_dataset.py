import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import soundfile as sf

# Constants
METADATA_PATH = "../data/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "../data/UrbanSound8K/audio"
OUTPUT_DIR = "../data/processed"
SPECTROGRAM_SIZE = (128, 128)  # Fixed size for all spectrograms
SAMPLE_RATE = 22050  # Standard sample rate
N_MELS = 128  # Number of mel bands
HOP_LENGTH = 512  # Hop length for STFT

def load_metadata(fold=1):
    """Load metadata for a specific fold."""
    df = pd.read_csv(METADATA_PATH)
    return df[df['fold'] == fold]

def create_mel_spectrogram(audio_path):
    """Create a mel spectrogram from an audio file."""
    # Load audio file
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # Create mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    return mel_spec_norm

def resize_spectrogram(spectrogram, target_size):
    """Resize or pad spectrogram to target size."""
    current_height, current_width = spectrogram.shape
    
    # If spectrogram is larger than target size, resize down
    if current_height > target_size[0] or current_width > target_size[1]:
        return librosa.util.fix_length(spectrogram, size=target_size[1], axis=1)[:target_size[0]]
    
    # If spectrogram is smaller, pad with zeros
    padded = np.zeros(target_size)
    padded[:current_height, :current_width] = spectrogram
    return padded

def load_metadata_all_folds():
    """Load metadata for all folds."""
    df = pd.read_csv(METADATA_PATH)
    return df

def process_all_folds():
    """Process all audio files in all folds and save combined data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata = load_metadata_all_folds()
    print(f"Processing {len(metadata)} files from all folds")

    X = []
    y = []
    # Use all classes in the dataset for mapping
    class_names = sorted(metadata['class'].unique())
    class_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing audio files"):
        audio_path = os.path.join(AUDIO_DIR, f"fold{row['fold']}", row['slice_file_name'])
        try:
            mel_spec = create_mel_spectrogram(audio_path)
            mel_spec_resized = resize_spectrogram(mel_spec, SPECTROGRAM_SIZE)
            X.append(mel_spec_resized)
            y.append(class_mapping[row['class']])
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            continue

    X = np.array(X)
    y = np.array(y)
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUTPUT_DIR, "class_mapping.npy"), class_mapping)

    print(f"\nProcessed data saved to {OUTPUT_DIR}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("\nClass mapping:")
    for class_name, idx in class_mapping.items():
        print(f"{class_name}: {idx}")

def main():
    process_all_folds()

if __name__ == "__main__":
    main() 