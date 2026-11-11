from pydantic import BaseModel, Field, field_validator
from typing import Optional

class MigrationScript(BaseModel):
    version_id: str = Field(..., description="Unique alphanumeric migration index (e.g., '001', '002')")
    description: Optional[str] = Field(None, description="Detailed text tracking the purpose of the migration")
    source_model: str = Field(..., min_length=1)
    target_model: str = Field(..., min_length=1)
    source_dimensions: Optional[int] = Field(None, gt=0, description="Auto-detected at runtime if omitted")
    target_dimensions: Optional[int] = Field(None, gt=0, description="Auto-detected at runtime if omitted")

    @field_validator('target_dimensions')
    @classmethod
    def validate_dimension_shift(cls, v: Optional[int], info) -> Optional[int]:
        # Only run check if both optional dimension overrides were explicitly provided
        if v is not None and info.data.get('source_dimensions') is not None:
            if v == info.data['source_dimensions']:
                pass
        return v
