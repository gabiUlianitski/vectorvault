import sqlite3
import numpy as np
import sqlite_vec
from sqlite_vec import serialize_float32
from sentence_transformers import SentenceTransformer
from vectorvault.engine import VectorProjectionEngine

SOURCE_MODEL_NAME = "all-MiniLM-L6-v2"   # 384 dimensions
TARGET_MODEL_NAME = "all-mpnet-base-v2"  # 768 dimensions

print("📥 Step 1: Loading embedding models...")
model_384 = SentenceTransformer(SOURCE_MODEL_NAME)
model_768 = SentenceTransformer(TARGET_MODEL_NAME)

print("⚙️ Step 2: Loading cached projection matrix or auto-calibrating from built-in anchors...")
engine = VectorProjectionEngine()
engine.get_or_train_matrix(
    source_name=SOURCE_MODEL_NAME,
    target_name=TARGET_MODEL_NAME,
    source_model=model_384,
    target_model=model_768
)

# Unseen customer records sitting in legacy 384D storage (NOT in DEFAULT_ANCHORS!)
unseen_customer_records = [
    "Quarterly revenue increased by eighteen percent across European enterprise accounts.",
    "Solar microgrids store excess daytime electricity in lithium iron phosphate batteries.",
    "Clinical trials evaluate drug safety and dosage efficacy across patient cohorts.",
    "Autonomous delivery drones navigate urban airspace using LiDAR obstacle avoidance."
]

print("📦 Step 3: Simulating legacy 384D customer vectors & migrating to empty 768D target DB...")
legacy_vectors_384 = model_384.encode(unseen_customer_records).tolist()

# Pure vector-to-vector migration (zero raw text passed to the target model!)
migrated_vectors_768 = engine.transform_vectors(legacy_vectors_384)

db = sqlite3.connect("vault_data.db")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

db.execute("DROP TABLE IF EXISTS legacy_storage_384d")
db.execute("CREATE VIRTUAL TABLE legacy_storage_384d USING vec0(embedding float[384])")

db.execute("DROP TABLE IF EXISTS migrated_storage_768d")
db.execute("CREATE VIRTUAL TABLE migrated_storage_768d USING vec0(embedding float[768])")

db.execute("DROP TABLE IF EXISTS record_metadata")
db.execute("CREATE TABLE record_metadata (rowid INTEGER PRIMARY KEY, label TEXT, orig_dim INT, new_dim INT)")

for i, text in enumerate(unseen_customer_records):
    row_id = i + 1
    db.execute("INSERT INTO legacy_storage_384d(rowid, embedding) VALUES (?, ?)", (row_id, serialize_float32(legacy_vectors_384[i])))
    db.execute("INSERT INTO migrated_storage_768d(rowid, embedding) VALUES (?, ?)", (row_id, serialize_float32(migrated_vectors_768[i])))
    db.execute("INSERT INTO record_metadata(rowid, label, orig_dim, new_dim) VALUES (?, ?, ?, ?)", (row_id, text, len(legacy_vectors_384[i]), len(migrated_vectors_768[i])))

db.commit()
db.close()

print("\n📊 Step 4: Out-of-Sample Semantic Equivalence Verification (384D vs 768D)")
def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

sim_384 = [cosine_sim(legacy_vectors_384[0], v) for v in legacy_vectors_384]
sim_768 = [cosine_sim(migrated_vectors_768[0], v) for v in migrated_vectors_768]

rank_384 = np.argsort(sim_384)[::-1].tolist()
rank_768 = np.argsort(sim_768)[::-1].tolist()

print(f"├── Source Space (384D) Search Ranking : {rank_384}")
print(f"├── Target Space (768D) Search Ranking : {rank_768}")
print(f"└── Unseen Data Identity Preserved?    : {rank_384 == rank_768} ✅")
