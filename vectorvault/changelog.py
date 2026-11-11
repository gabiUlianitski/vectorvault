import click

# The foundational DDL execution query that sets up the tracking state
INIT_CHANGELOG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS vector_vault_changelog (
    id SERIAL PRIMARY KEY,
    version_id VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    source_model VARCHAR(100) NOT NULL,
    target_model VARCHAR(100) NOT NULL,
    source_dimensions INTEGER NOT NULL,
    target_dimensions INTEGER NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'COMPLETED'
);
"""

def initialize_metadata_ledger(conn) -> bool:
    """
    Executes transactional DDL to set up the Liquibase-style 
    tracking table inside the database.
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(INIT_CHANGELOG_TABLE_SQL)
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        click.echo(f"❌ Changelog Initialization Error: {e}", err=True)
        return False
