################################################################################
# This file documents the exact layout of the returned .JSON files for the 
# profiler, including the expected fields and their types, as well as the
# cleaning outputs from the LLM, and any tools needed by the Agent.
################################################################################

# Imports
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

################################################################################
# LLM Action Enumerations:
# These enumerations control what actions the LLM can take to "clean" the input
# dataset.
################################################################################

OperationType = Literal[
    "DROP_COLUMN",
    "DROP_ROWS", 
    "IMPUTE_MEAN",
    "IMPUTE_MEDIAN",
    "IMPUTE_MODE",
    "IMPUTE_CONSTANT",
    "RENAME_COLUMN",
    "CAST_TYPE",
    "StandardScaler",
    "MinMaxScaler",
    "OneHotEncode",
    "CUSTOM_CODE" # For any custom cleaning code not covered by the above operations.
    # We can add more operations as needed, but let's start here for now.
]

################################################################################
# Schema Definitions for Profiler to Agent:
# These schemas define the exact structure of the JSON files exchanged between
# the Profiler and the Semantic Agent.
################################################################################

# Schema definitions for each column
class ColumnProfile(BaseModel):
    name: str = Field(..., description="The name of the column in the dataframe.")
    inferred_type: str = Field(..., description="The inferred data type (e.g., 'Numeric', 'Categorical', 'Datetime').")
    
    # Completeness & Uniqueness
    completeness: float = Field(..., description="Ratio of non-null values (0.0 to 1.0). 1.0 means no missing values.")
    unique_factor: float = Field(..., description="Ratio of unique values to total rows. Low (<0.01) implies categorical.")
    
    # Statistical Shape (Critical for Semantic Detection)
    min_value: Optional[float] = Field(None, description="Minimum value. Look for -1, 0, or outliers.")
    max_value: Optional[float] = Field(None, description="Maximum value. Look for 999, 9999, or future years.")
    mean: Optional[float] = Field(None, description="Arithmetic mean.")
    skewness: Optional[float] = Field(None, description="Fisher-Pearson skewness. >1 or <-1 implies heavy skew.")

    # Numeric Extra Signals (Optional / Non-breaking)
    median: Optional[float] = Field(None, description="Median value (robust central tendency).")
    zero_count: Optional[int] = Field(None, description="Count of exact zeros (useful for sentinels/encoding).")
    negative_count: Optional[int] = Field(None, description="Count of negative values.")
    inf_nan_count: Optional[int] = Field(None, description="Count of inf/-inf/NaN after numeric coercion.")

    # Datetime Signals (Optional / Non-breaking)
    earliest_date: Optional[str] = Field(None, description="Earliest date (ISO string if possible).")
    latest_date: Optional[str] = Field(None, description="Latest date (ISO string if possible).")
    datetime_format_consistency: Optional[float] = Field(
        None, description="0–1 ratio: how consistently values match the dominant datetime format."
    )

    # String/Categorical Signals (Optional / Non-breaking)
    regex_format_consistency: Optional[float] = Field(
        None, description="0–1 ratio: how consistently values match the dominant regex-like pattern."
    )
    dominant_pattern: Optional[str] = Field(
        None, description="Short signature of dominant pattern (kept compact for LLM tokens)."
    )
    
    # Contextual Samples (The "Fingerprint")
    top_frequent_values: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Top 5 most frequent values and their counts. Used to detect sentinels like '999' or 'N/A'."
    )
    
    # Profiler Warnings
    semantic_warnings: List[str] = Field(
        default_factory=list,
        description="List of strings flagging issues (e.g., 'High Cardinality', 'Constant Value', 'Possible PII')."
    )

# Schema definition for the entire dataset
class DatasetProfile(BaseModel):
    row_count: int = Field(..., description="Total number of rows in the raw dataset.")
    columns: List[ColumnProfile] = Field(..., description="List of profiles for each column.")


################################################################################
# Schema Definitions for Agent to Cleaner:
# These schemas define the exact structure of the JSON files exchanged between
# the Semantic Agent and the cleaning sandbox.
#################################################################################

class CleaningStep(BaseModel):
    step_id: int = Field(..., description="The execution order (1-indexed).")
    operation: OperationType = Field(..., description="The category of operation being performed.")
    target_column: Optional[str] = Field(None, description="The specific column being modified (if applicable).")
    
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the operation (e.g., {'value': 0} for IMPUTE_CONSTANT)."
    )
    
    justification: str = Field(..., description="Semantic reasoning: WHY are we doing this? (e.g., 'Detected -1 in Age column, likely sentinel').")
    
    python_code: str = Field(
        ..., 
        description="The actual executable Python code snippet using 'df'. E.g., `df = df[df['age'] > 0]`"
    )

class CleaningRecipe(BaseModel):
    """The full plan returned by the Semantic Agent."""
    steps: List[CleaningStep]

################################################################################
# Schema Definitions for Audit Trail:
# Tracks the actual history of what was executed in the Sandbox to retrain
# context.
################################################################################

class CleaningLogEntry(BaseModel):
    timestamp: str = Field(..., description="ISO 8601 timestamp of when the step was executed.")
    step_id: int = Field(..., description="The ID of the step from the original plan.")
    operation: str = Field(..., description="The operation performed.")
    justification: str = Field(..., description="The reasoning behind the operation.")
    code_executed: str = Field(..., description="The actual Python code that was run.")
    status: Literal["SUCCESS", "FAILED"] = Field(..., description="Whether the execution succeeded or failed.")


################################################################################
# Schema Definitions for Tools:
# These schemas define the exact structure of the JSON files exchanged between
# the Semantic Agent and the Data Tools for tool inputs/outputs.
################################################################################

class InspectRowsInput(BaseModel):
    query: str = Field(..., description="Pandas query string to filter rows (e.g., 'age < 0').")
    limit: int = Field(5, description="Number of rows to return (default 5). Keep this small to save tokens.")