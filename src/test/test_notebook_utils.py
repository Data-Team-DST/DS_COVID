"""
Tests for the notebook utilities module.
"""

import numpy as np
import pytest

from utils import (
    build_custom_cnn,
    build_transfer_learning_model,
    compile_model,
    compute_class_weights,
    create_callbacks,
    create_preprocessing_pipeline,
    evaluate_model,
    get_preprocessing_function,
    load_dataset,
    plot_confusion_matrix,
    plot_training_curves,
    prepare_train_val_test_split,
    run_gradcam_analysis,
    select_sample_images,
    setup_interpretability,
    train_model,
    unfreeze_top_layers,
)


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    rng = np.random.default_rng()
    X = rng.random((100, 224, 224, 3))
    y = rng.integers(0, 2, 100)
    return X, y


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=2)
    return model


# Test Data Loading & Preprocessing
def test_load_dataset():
    """Test dataset loading function."""
    with pytest.raises(ValueError):
        load_dataset("")  # Should raise error for invalid path


def test_create_preprocessing_pipeline():
    """Test preprocessing pipeline creation."""
    pipeline = create_preprocessing_pipeline()
    assert callable(pipeline)


def test_prepare_train_val_test_split(sample_data):
    """Test data splitting function."""
    X, y = sample_data
    splits = prepare_train_val_test_split(X, y, val_size=0.2, test_size=0.2)
    assert len(splits) == 6  # Should return 6 arrays


def test_compute_class_weights(sample_data):
    """Test class weights computation."""
    _, y = sample_data
    weights = compute_class_weights(y)
    assert isinstance(weights, dict)


# Test Model Building
def test_build_custom_cnn():
    """Test custom CNN model building."""
    model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=2)
    assert model is not None


def test_compile_model(sample_model):
    """Test model compilation."""
    compiled_model = compile_model(sample_model)
    assert compiled_model.optimizer is not None


def test_create_callbacks():
    """Test callback creation."""
    callbacks = create_callbacks()
    assert len(callbacks) > 0


def test_build_transfer_learning_model():
    """Test transfer learning model building."""
    input_shape = (224, 224, 3)
    model = build_transfer_learning_model(input_shape=input_shape, num_classes=2)
    assert model is not None


def test_unfreeze_top_layers(sample_model):
    """Test layer unfreezing."""
    unfrozen_model = unfreeze_top_layers(sample_model, num_layers=5)
    assert unfrozen_model is not None


# Test Training & Evaluation
def test_train_model(sample_model, sample_data):
    """Test model training."""
    X, y = sample_data
    history = train_model(sample_model, X, y, epochs=1, batch_size=32)
    assert history is not None


def test_evaluate_model(sample_model, sample_data):
    """Test model evaluation."""
    X, y = sample_data
    metrics = evaluate_model(sample_model, X, y)
    assert isinstance(metrics, dict)


# Test Visualization
def test_plot_training_curves():
    """Test training curves plotting."""
    history = {"loss": [0.5, 0.3], "val_loss": [0.6, 0.4]}
    fig = plot_training_curves(history)
    assert fig is not None


def test_plot_confusion_matrix():
    """Test confusion matrix plotting."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    fig = plot_confusion_matrix(y_true, y_pred)
    assert fig is not None


# Test Interpretability
def test_setup_interpretability(sample_model):
    """Test interpretability setup."""
    setup = setup_interpretability(sample_model)
    assert setup is not None


def test_run_gradcam_analysis(sample_model, sample_data):
    """Test Grad-CAM analysis."""
    X, _ = sample_data
    heatmap = run_gradcam_analysis(sample_model, X[0])
    assert heatmap is not None


def test_select_sample_images(sample_data):
    """Test sample image selection."""
    X, y = sample_data
    samples = select_sample_images(X, y, num_samples=5)
    assert len(samples) == 5


def test_get_preprocessing_function():
    """Test preprocessing function retrieval."""
    preprocess_fn = get_preprocessing_function()
    assert callable(preprocess_fn)
