"""Training utilities with MLflow integration."""

import time

import mlflow
import tensorflow as tf

from cv_toolkit.config import Config, PhaseConfig

def get_callbacks(patience=5, min_lr=1e-6):
    """Create standard training callbacks."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=min_lr,
            verbose=1,
        ),
    ]

def compile_model(model, learning_rate):
    """Compile model with Adam optimizer."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

def train_model(model, train_ds, val_ds, phase_config: PhaseConfig, callbacks=None):
    """Train model for one phase."""
    compile_model(model, phase_config.learning_rate)
    
    if callbacks is None:
        callbacks = get_callbacks()
    
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase_config.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    training_time = time.time() - start_time
    
    print(f"Training time: {training_time / 60:.1f} min")
    return history, training_time

def train_with_mlflow(model, train_ds, val_ds, config: Config, phase_config: PhaseConfig, run_name: str):
    """Train model with MLflow experiment tracking."""
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            "learning_rate": phase_config.learning_rate,
            "epochs": phase_config.epochs,
            "batch_size": config.data.batch_size,
            "model": config.model.base_model,
        })
        
        # Train
        history, training_time = train_model(model, train_ds, val_ds, phase_config)
        
        # Log metrics
        mlflow.log_metrics({
            "best_val_accuracy": max(history.history["val_accuracy"]),
            "final_val_loss": history.history["val_loss"][-1],
            "training_time_min": training_time / 60,
        })
        
        return history, training_time
