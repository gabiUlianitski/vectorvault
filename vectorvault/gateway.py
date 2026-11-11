from typing import List, Dict, Any, Optional
from vectorvault.models import MigrationScript

class VectorSchemaCircuitBreaker:
    def __init__(self, expected_dimensions: int, active_model_name: str):
        self.expected_dimensions = expected_dimensions
        self.active_model_name = active_model_name
        self.is_tripped = False

    def inspect_query_vector(self, query_vector: List[float]) -> Dict[str, Any]:
        """
        Inspects an incoming search query vector before it strikes the database tier.
        Acts as an inline firewall to intercept dimensional shape mismatches.
        """
        actual_dimensions = len(query_vector)

        if actual_dimensions != self.expected_dimensions:
            self.is_tripped = True
            return {
                "allowed": False,
                "reason": "DIMENSION_MISMATCH",
                "message": (
                    f"🚨 VectorVault Circuit Breaker Tripped! Inbound query dimension ({actual_dimensions}) "
                    f"does not align with active database schema expectations ({self.expected_dimensions}). "
                    f"Active production model is configured to: '{self.active_model_name}'."
                ),
                "action_required": "Deploy a VectorVault migration script or re-route via VectorProjectionEngine."
            }

        return {"allowed": True, "reason": "CLEAN", "message": "Query vector dimensions are synchronized."}
