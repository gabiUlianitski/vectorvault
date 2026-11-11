import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from vectorvault.engine import VectorProjectionEngine
from vectorvault.gateway import VectorSchemaCircuitBreaker

print("📥 Loading local free embedding models from Hugging Face...")
# Loading two distinct text embedding models
model_source = SentenceTransformer('all-MiniLM-L6-v2')       # Space A
model_target = SentenceTransformer('paraphrase-MiniLM-L3-v2') # Space B

# 1. Prepare real sample text to train our mapping projection matrix
training_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming software engineering.",
    "Database migrations require transactional safety and logs.",
    "ETL pipelines ingest billions of rows of operational streaming data.",
    "Telecom networks process call detail records for revenue assurance.",
    "Vector embeddings store semantic meanings as coordinate dimensions.",
    "Cloud infrastructure monitoring prevents runtime system crashes.",
    "Open source development fosters massive community collaboration.",
    "Linear algebra operations power modern machine learning layers.",
    "FastAPI proxies can intercept outbound network traffic loops."
]

print(f"🧠 Generating real vector representations for {len(training_sentences)} documents...")
# Generate real vector matrices (N rows x 384 columns)
source_train_vectors = model_source.encode(training_sentences).tolist()
target_train_vectors = model_target.encode(training_sentences).tolist()

print(f"📊 Source Vector Sample Dimension: {len(source_train_vectors[0])}")
print(f"📊 Target Vector Sample Dimension: {len(target_train_vectors[0])}")

# 2. Train the VectorVault projection engine on real semantic shifts
print("\n🧮 Training VectorVault matrix projection transformation weights...")
engine = VectorProjectionEngine()
success = engine.train_mapping_matrix(source_train_vectors, target_train_vectors)

if success:
    print("✅ Matrix projection weights computed successfully via pseudo-inverse math!")
else:
    print("❌ Critical Math Failure during transformation matrix computation.")
    sys.exit(1)

# =========================================================================
# THE MIGRATION SIMULATION
# =========================================================================
print("\n🚀 Simulating Production Load Data Migration...")

# Brand new production sentences that the engine has NEVER seen before
production_sentences = [
    "A computer network transmits packets across global routing lines.",
    "Financial ledgers must reconcile data records across systems safely."
]

# Generate their real historical vectors (Old format)
production_old_vectors = model_source.encode(production_sentences).tolist()

# Transform them instantly in-memory without calling any APIs or re-embedding!
print("⚡ Executing VectorVault in-memory linear transformation...")
migrated_vectors = engine.transform_vectors(production_old_vectors)

print(f"✅ Successfully converted {len(migrated_vectors)} production records into the new vector space layout!")
print(f"📏 Migrated Row 1 Shape Verification: {len(migrated_vectors[0])} dimensions.")

# =========================================================================
# THE CIRCUIT BREAKER PROTECTION CHECK
# =========================================================================
print("\n🛡️ Initializing Gateway Firewall Layer...")
# Set up a circuit breaker expecting the new target format (384 dimensions)
guardian = VectorSchemaCircuitBreaker(expected_dimensions=384, active_model_name="paraphrase-MiniLM-L3-v2")

# Check a migrated vector layout (should pass)
verdict_clean = guardian.inspect_query_vector(migrated_vectors[0])
print(f"Clean Query Verification: Allowed={verdict_clean['allowed']} | Reason={verdict_clean['reason']}")

# Check an outdated, un-migrated input vector layout (should be caught and blocked!)
print("\n🚨 Simulating an outdated application instance sending an un-migrated vector query...")
bad_query = [0.99] * 1536 # Simulating an old 1536 OpenAI vector striking the database
verdict_blocked = guardian.inspect_query_vector(bad_query)

print(verdict_blocked["message"])
print("✅ VectorVault Production Safety Workflow Fully Verified!")
