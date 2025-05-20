import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Constants
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LR = 0.0001
INPUT_SHAPE = (128, 128, 1)
NUM_CLASSES = 10

def load_data():
    """Load and preprocess the data"""
    print("Loading preprocessed data...")
    
    # Load the combined dataset
    X = np.load('../data/processed/X.npy')
    y = np.load('../data/processed/y.npy')
    class_mapping = np.load('../data/processed/class_mapping.npy', allow_pickle=True).item()
    
    # Reshape X for CNN input
    X = X.reshape(-1, 128, 128, 1)
    
    # Convert y to one-hot encoding
    y = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("\nClass mapping:")
    for class_name, idx in class_mapping.items():
        print(f"{class_name}: {idx}")
    
    return X, y, class_mapping

def build_model():
    """Build a deeper CNN model with more filters"""
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=INPUT_SHAPE),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Conv Block
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Conv Block
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Fourth Conv Block
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_data_generators(X_train, X_val, y_train, y_val):
    """Create data generators with augmentation for training and validation"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        zoom_range=0.2,
        shear_range=0.2
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    return train_generator, val_generator

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save plot
    plot_path = os.path.join('models', 'plots', 'training_history.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved to {plot_path}")

def main():
    # Create necessary directories
    os.makedirs('../models/plots', exist_ok=True)
    
    # Load data
    X, y, class_mapping = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Compute class weights
    y_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print("\nClass weights:")
    for class_name, idx in class_mapping.items():
        print(f"{class_name}: {class_weight_dict[idx]:.2f}")
    
    # Create data generators
    train_generator, val_generator = create_data_generators(X_train, X_val, y_train, y_val)
    
    # Build model
    model = build_model()
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            '../models/noise_classifier.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main() 