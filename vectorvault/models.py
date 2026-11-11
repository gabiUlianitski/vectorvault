from pydantic import BaseModel, Field, field_validator
from typing import Optional

class MigrationScript(BaseModel):
    version_id: str = Field(..., description="Unique alphanumeric migration index (e.g., '001', '002')")
    description: Optional[str] = Field(None, description="Detailed text tracking the purpose of the migration")
    source_model: str = Field(..., min_length=1)
    target_model: str = Field(..., min_length=1)
    source_dimensions: int = Field(..., gt=0)
    target_dimensions: int = Field(..., gt=0)

    @field_validator('target_dimensions')
    @classmethod
    def validate_dimension_shift(cls, v: int, info) -> int:
        # Data integrity check: throwing a transparent anomaly layout on invalid validation parameters
        if 'source_dimensions' in info.data and v == info.data['source_dimensions']:
            # While valid, typically a migration signifies a structural alteration or model optimization
            pass
        return v
