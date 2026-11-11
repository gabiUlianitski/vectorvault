import sqlite3
import struct
import sqlite_vec
from sqlite_vec import serialize_float32
from sentence_transformers import SentenceTransformer
from vectorvault.engine import VectorProjectionEngine


class VectorVaultMigrator:
    """
    End-to-end DataOps orchestrator that reads vectors from a Source Database,
    auto-detects dimensions, converts them across models via VectorProjectionEngine,
    and writes them into an empty Target Database.
    """

    def __init__(self, source_model_name: str, target_model_name: str):
        self.source_model_name = source_model_name
        self.target_model_name = target_model_name
        self.engine = VectorProjectionEngine()

    def _prepare_engine(self):
        if not self.engine.load_matrix(self.source_model_name, self.target_model_name):
            src_model = SentenceTransformer(self.source_model_name)
            tgt_model = SentenceTransformer(self.target_model_name)
            self.engine.get_or_train_matrix(
                source_name=self.source_model_name,
                target_name=self.target_model_name,
                source_model=src_model,
                target_model=tgt_model
            )

    def migrate_sqlite(
        self,
        source_db_path: str,
        source_table: str,
        target_db_path: str,
        target_table: str
    ) -> dict:
        """
        Reads existing vectors from Source SQLite DB, auto-detects source & target
        dimensions, and writes converted vectors into an empty Target SQLite DB.
        """
        self._prepare_engine()

        # 1. Read raw vector BLOBs from Source SQLite DB
        src_conn = sqlite3.connect(source_db_path)
        src_conn.enable_load_extension(True)
        sqlite_vec.load(src_conn)
        src_conn.enable_load_extension(False)

        rows = src_conn.execute(f"SELECT rowid, embedding FROM {source_table}").fetchall()
        src_conn.close()

        if not rows:
            return {"count": 0, "source_dim": 0, "target_dim": 0}

        # Auto-detect source dimension from byte size (float32 = 4 bytes per float)
        source_dim = len(rows[0][1]) // 4
        row_ids = [r[0] for r in rows]
        legacy_vectors = [
            list(struct.unpack(f"{source_dim}f", r[1])) for r in rows
        ]

        # 2. Convert vectors mathematically
        migrated_vectors = self.engine.transform_vectors(legacy_vectors)

        # Auto-detect target dimension from converted output
        target_dim = len(migrated_vectors[0])

        # 3. Write converted vectors into Target SQLite DB
        tgt_conn = sqlite3.connect(target_db_path)
        tgt_conn.enable_load_extension(True)
        sqlite_vec.load(tgt_conn)
        tgt_conn.enable_load_extension(False)

        tgt_conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {target_table} USING vec0(embedding float[{target_dim}])"
        )

        for row_id, new_vec in zip(row_ids, migrated_vectors):
            tgt_conn.execute(
                f"INSERT OR REPLACE INTO {target_table}(rowid, embedding) VALUES (?, ?)",
                (row_id, serialize_float32(new_vec))
            )

        tgt_conn.commit()
        tgt_conn.close()

        return {
            "count": len(migrated_vectors),
            "source_dim": source_dim,
            "target_dim": target_dim
        }
