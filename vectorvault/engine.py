import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Any

class VectorProjectionEngine:
    DEFAULT_ANCHORS: List[str] = [
        "Database migrations require transactional safety, rollback plans, and audit logs.",
        "Distributed systems balance consistency, availability, and network partition tolerance.",
        "Financial ledgers reconcile daily settlement transactions across multi-currency accounts.",
        "Machine learning models optimize objective loss functions via stochastic gradient descent.",
        "Cloud infrastructure scales containerized microservices automatically under heavy traffic.",
        "Telecom billing engines process streaming call detail records for revenue assurance.",
        "Cybersecurity firewalls inspect inbound network packets for anomalous payloads.",
        "Healthcare electronic records enforce strict patient privacy and access controls.",
        "Supply chain logistics track global inventory shipments across maritime ports.",
        "Software engineering teams automate continuous integration and deployment pipelines.",
        "Relational databases enforce referential integrity through primary and foreign keys.",
        "Vector databases accelerate nearest-neighbor semantic search using HNSW indexes.",
        "Natural language processing converts unstructured text into dense semantic embeddings.",
        "Enterprise data warehouses aggregate historical metrics for executive analytics.",
        "API gateways manage authentication, rate limiting, and upstream request routing.",
        "Kubernetes clusters orchestrate container scheduling across distributed worker nodes.",
        "Payment processors authorize credit card transactions and detect fraudulent patterns.",
        "E-commerce recommendation systems rank products based on user behavioral signals.",
        "Real-time event streaming platforms ingest millions of messages per second via Kafka.",
        "Cryptographic hash functions verify data immutability and detect record corruption."
    ]

    def __init__(self, cache_dir: str = "~/.vectorvault/matrices"):
        self.projection_matrix: Optional[np.ndarray] = None
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def train_mapping_matrix(self, source_samples: List[List[float]], target_samples: List[List[float]]) -> bool:
        """
        Learns the mathematical transformation matrix between two vector spaces
        using Ordinary Least Squares (OLS) linear projection.
        
        Args:
            source_samples: Small sample matrix from the old embedding model (N x D1)
            target_samples: Same text samples from the new embedding model (N x D2)
        """
        X = np.array(source_samples)
        Y = np.array(target_samples)
        
        if X.shape[0] != Y.shape[0]:
            raise ValueError("❌ Matrix Alignment Failure: Source and Target sample row lengths must match.")

        try:
            # Linear Projection Formula: W = (X^T * X)^(-1) * X^T * Y
            # Using numpy's robust pseudo-inverse (pinv) to handle potential singularity issues cleanly
            X_pseudo_inverse = np.linalg.pinv(X)
            self.projection_matrix = np.dot(X_pseudo_inverse, Y)
            return True
        except Exception as e:
            self.projection_matrix = None
            return False

    def calibrate_from_models(
        self,
        source_model: Any,
        target_model: Any,
        custom_anchors: Optional[List[str]] = None
    ) -> bool:
        """
        Automatically generates calibration pairs from built-in or custom anchor
        sentences when the target database is empty, then trains the projection matrix.
        """
        anchors = custom_anchors if custom_anchors else self.DEFAULT_ANCHORS
        source_samples = source_model.encode(anchors).tolist()
        target_samples = target_model.encode(anchors).tolist()
        return self.train_mapping_matrix(source_samples, target_samples)

    def _get_matrix_path(self, source_name: str, target_name: str) -> Path:
        safe_source = source_name.replace("/", "_").replace(":", "_")
        safe_target = target_name.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_source}_to_{safe_target}.npy"

    def save_matrix(self, source_name: str, target_name: str) -> str:
        """Persists the trained projection matrix to disk (.npy) for instant reuse."""
        if self.projection_matrix is None:
            raise RuntimeError("❌ Save Blocked: Cannot save projection matrix before training.")
        filepath = self._get_matrix_path(source_name, target_name)
        np.save(filepath, self.projection_matrix)
        return str(filepath)

    def load_matrix(self, source_name: str, target_name: str) -> bool:
        """Loads a pre-trained projection matrix from disk if cached."""
        filepath = self._get_matrix_path(source_name, target_name)
        if filepath.exists():
            self.projection_matrix = np.load(filepath)
            return True
        return False

    def get_or_train_matrix(
        self,
        source_name: str,
        target_name: str,
        source_model: Any = None,
        target_model: Any = None,
        custom_anchors: Optional[List[str]] = None
    ) -> bool:
        """
        Loads a cached projection matrix if available; otherwise calibrates using
        the provided models and caches the matrix for future migrations.
        """
        if self.load_matrix(source_name, target_name):
            return True

        if source_model is None or target_model is None:
            raise ValueError(
                f"❌ Cache Miss: No pre-trained matrix found for '{source_name}' -> '{target_name}', "
                "and model instances were not provided for calibration."
            )

        if self.calibrate_from_models(source_model, target_model, custom_anchors):
            self.save_matrix(source_name, target_name)
            return True
        return False

    def transform_vectors(self, source_vectors: List[List[float]]) -> List[List[float]]:
        """
        Transforms full production vector datasets using the calculated projection matrix.
        Performs ultra-fast, local matrix multiplication in milliseconds.
        """
        if self.projection_matrix is None:
            raise RuntimeError("❌ Execution Blocked: Cannot transform vectors before training the projection matrix.")
            
        X = np.array(source_vectors)
        
        # Multiply production vector matrix by our projection transformation weights
        transformed_matrix = np.dot(X, self.projection_matrix)
        
        # Convert back to native python lists for database driver compliance
        return transformed_matrix.tolist()
