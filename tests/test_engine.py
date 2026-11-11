import pytest
import numpy as np
from vectorvault.engine import VectorProjectionEngine

def test_vector_space_transformation_lifecycle():
    engine = VectorProjectionEngine()
    
    # 1. Simulate 50 small sample documents embedded across two different sizes
    # Model A (Old: 1536 dimensions) vs. Model B (New: 768 dimensions)
    np.random.seed(42)  # Locking randomness for deterministic validation
    mock_source_samples = np.random.randn(50, 1536).tolist()
    mock_target_samples = np.random.randn(50, 768).tolist()
    
    # 2. Train the transformation mapping engine
    success = engine.train_mapping_matrix(mock_source_samples, mock_target_samples)
    assert success is True
    assert engine.projection_matrix.shape == (1536, 768)
    
    # 3. Simulate a production load of 200 old vectors that need instant migration
    production_old_vectors = np.random.randn(200, 1536).tolist()
    
    # 4. Transform the dimensions instantly in-process
    migrated_vectors = engine.transform_vectors(production_old_vectors)
    
    # 5. Assert shape compliance boundaries
    assert len(migrated_vectors) == 200
    assert len(migrated_vectors[0]) == 768
    print("\n✅ Mathematical Vector Space Transformation Lifecycle Passed!")
