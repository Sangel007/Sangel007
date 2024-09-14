import tensorflow as tf
from model import create_hybrid_model
from data_loader import load_data

# Function to train the model
def train_model(dataset_dir, epochs=20, batch_size=32):
    # Load data
    train_data, val_data = load_data(dataset_dir, batch_size=batch_size)

    # Create hybrid model
    hybrid_model = create_hybrid_model()

    # Compile the model
    hybrid_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    history = hybrid_model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data
    )

    # Save the trained model
    hybrid_model.save('hybrid_age_detection_model.h5')

    return history

