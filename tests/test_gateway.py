import pytest
from vectorvault.gateway import VectorSchemaCircuitBreaker

def test_gateway_circuit_breaker_interception_lifecycle():
    # 1. Establish a live production guard expecting Llama3 768-dimension arrays
    guardian = VectorSchemaCircuitBreaker(expected_dimensions=768, active_model_name="llama3-embedding")
    
    # 2. Simulate a perfectly clean query vector coming from the application layer
    clean_query = [0.1] * 768
    verdict_clean = guardian.inspect_query_vector(clean_query)
    assert verdict_clean["allowed"] is True
    assert guardian.is_tripped is False
    
    # 3. Simulate an outdated app instance trying to query using an old 1536-dimension OpenAI vector
    broken_query = [0.5] * 1536
    verdict_broken = guardian.inspect_query_vector(broken_query)
    
    # 4. Assert that the circuit breaker actively intercepted the bad math query
    assert verdict_broken["allowed"] is False
    assert verdict_broken["reason"] == "DIMENSION_MISMATCH"
    assert "Circuit Breaker Tripped" in verdict_broken["message"]
    assert guardian.is_tripped is True
    print("\n✅ Gateway Schema Circuit Breaker Interception Verified!")
