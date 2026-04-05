################################################################################
# This file documents the exact layout of the returned .JSON files for the 
# profiler, including the expected fields and their types, as well as the
# cleaning outputs from the LLM, and any tools needed by the Agent.
################################################################################

# Imports
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator

################################################################################
# LLM Action Enumerations:
# These enumerations control what actions the LLM can take to "clean" the input
# dataset.
#
# NOTE: StandardScaler and MinMaxScaler have been REMOVED. These operations fit
# statistics (mean/std/min/max) on the full dataset. If applied before AutoGluon
# splits into train/test, test set statistics leak into training. AutoGluon 
# handles its own internal preprocessing, so these are both dangerous and 
# redundant. See Issue #25.
################################################################################

OperationType = Literal[
    "DROP_COLUMN",
    "DROP_ROWS",
    "RENAME_COLUMN",
    "CAST_TYPE",
    "CUSTOM_CODE"
]


################################################################################
# Schema Definitions for Profiler to Agent:
# These schemas define the exact structure of the JSON files exchanged between
# the Profiler and the Semantic Agent.
################################################################################

class ColumnProfile(BaseModel):
    name: str = Field(..., description="The name of the column in the dataframe.")
    inferred_type: str = Field(..., description="The inferred data type (e.g., 'Numeric', 'Categorical', 'Datetime').")

    # Optional profiler gap-fix fields (optional / non-breaking)
    actual_dtype: Optional[str] = Field(
        None, description="Actual pandas dtype for the column (e.g., 'object', 'int64', 'float64', 'datetime64[ns]')."
    )
    type_mismatch: Optional[bool] = Field(
        None, description="True when inferred_type suggests Numeric/Datetime but actual_dtype is object/string-like."
    )
    coercion_failure_count: Optional[int] = Field(
        None, description="For inferred Numeric/Datetime: count of values that fail coercion (excluding natural NaN)."
    )
    coercion_failure_samples: Optional[List[str]] = Field(
        None, description="Sample raw values that failed coercion (3–5), to help codegen write correct cleaning logic."
    )
    random_sample_values: Optional[List[Any]] = Field(
        None, description="Random sample of 3–5 values (JSON-safe) to expose edge cases beyond top frequent values."
    )
    q1: Optional[float] = Field(None, description="25th percentile (Q1) for numeric columns.")
    q3: Optional[float] = Field(None, description="75th percentile (Q3) for numeric columns.")
    datetime_format_samples: Optional[List[str]] = Field(
        None, description="Sample competing datetime string formats when consistency is low (3–5 examples)."
    )
    
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
    inf_nan_count: Optional[int] = Field(None, description="Count of inf/-inf values and coercion failures (excludes natural NaN — AutoGluon handles missing data natively).")

    # Datetime Signals (Optional / Non-breaking)
    earliest_date: Optional[str] = Field(None, description="Earliest date (ISO string if possible).")
    latest_date: Optional[str] = Field(None, description="Latest date (ISO string if possible).")
    datetime_format_consistency: Optional[float] = Field(
        None, description="0–1 ratio: how consistently values match the dominant datetime format."
    )
    future_date_count: Optional[int] = Field(
        None, description="Count of date values after today's date. Non-zero indicates likely data entry errors."
    )

    # String/Categorical Signals (Optional / Non-breaking)
    regex_format_consistency: Optional[float] = Field(
        None, description="0–1 ratio: how consistently values match the dominant regex-like pattern."
    )
    dominant_pattern: Optional[str] = Field(
        None, description="Short signature of dominant pattern (kept compact for LLM tokens)."
    )

    # Optional: compact ydata-profiling enrichment (whitelisted keys only)
    ydata_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional dict of extra ydata-derived metrics (kept compact; do NOT store raw ydata JSON)."
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

class AlgorithmicQualityScore(BaseModel):
    """DEPRECATED — kept for backward compatibility during transition.

    Use QualityAssessment instead, which combines structural checks with
    LLM-based semantic evaluation.
    """
    overall: float = Field(..., description="Weighted composite score 0.0 (unusable) to 1.0 (pristine).")
    completeness: float = Field(..., description="Average completeness across all columns. Informational only — not weighted in overall score (AutoGluon handles missing features).")
    target_integrity: Optional[float] = Field(
        None, description="Target column health: completeness minus penalties for inf/nan/sentinel. "
                          "None if target_column not specified."
    )
    value_plausibility: float = Field(
        ..., description="Penalizes heaping/template patterns where a single value dominates a numeric column."
    )
    structural_integrity: float = Field(
        ..., description="Fraction of numeric columns free of inf/nan values."
    )
    flags: List[str] = Field(
        default_factory=list,
        description="Specific issues detected algorithmically (e.g., 'heaping: systolic_bp (120 in 73% of rows)')."
    )


################################################################################
# Quality Assessment (Three-Tier System)
#
# Replaces AlgorithmicQualityScore with:
#   Tier 1: Structural checks (deterministic, free)
#   Tier 2: Statistical anomaly summary (deterministic, cheap)
#   Tier 3: LLM quality assessor agent (semantic, configurable cost)
################################################################################

class ColumnAnomalySummary(BaseModel):
    """Per-column anomaly statistics from Tier 2 analysis."""
    column: str = Field(..., description="Column name.")
    outlier_count: int = Field(0, description="Number of statistical outliers detected.")
    outlier_direction: str = Field(
        "none", description="Direction of outliers: 'high', 'low', 'both', or 'none'."
    )
    rare_categories: Optional[List[str]] = Field(
        None, description="Rare category values (< 1% frequency). None for numeric columns."
    )


class AnomalySummary(BaseModel):
    """Tier 2 output: statistical anomaly summary across all columns."""
    column_summaries: List[ColumnAnomalySummary] = Field(
        default_factory=list, description="Per-column anomaly statistics."
    )
    total_rows: int = Field(0, description="Total rows in the dataset.")
    total_anomalous_rows: int = Field(
        0, description="Number of rows with at least one anomalous value."
    )


class LLMQualityAssessment(BaseModel):
    """Tier 3 output: structured quality assessment from the LLM assessor agent."""
    score: float = Field(
        ..., description="Quality score 0.0 (unusable) to 1.0 (clean). "
                         "Based on the LLM's semantic evaluation of the data."
    )
    recommendation: Literal["continue_cleaning", "stop_cleaning"] = Field(
        ..., description="Whether the data is clean enough to proceed to modeling."
    )
    reasoning: str = Field(
        ..., description="Explanation of the assessment — what the LLM checked and what it found."
    )
    residual_issues: List[str] = Field(
        default_factory=list,
        description="Specific remaining issues. Passed to the next investigator pass if continuing."
    )
    false_positives: List[str] = Field(
        default_factory=list,
        description="Anomaly flags that the LLM determined are legitimate data, not errors."
    )


class QualityAssessment(BaseModel):
    """Combined quality assessment from all three tiers."""
    structural_score: float = Field(
        ..., description="Tier 1: deterministic structural score (0.0-1.0). "
                         "Based on completeness and inf/nan presence."
    )
    anomaly_summary: AnomalySummary = Field(
        ..., description="Tier 2: statistical anomaly summary across columns."
    )
    llm_assessment: Optional[LLMQualityAssessment] = Field(
        None, description="Tier 3: LLM quality assessor's semantic evaluation. "
                          "None if LLM assessment was skipped (e.g., structural hard blocker)."
    )
    recommendation: Literal["continue_cleaning", "stop_cleaning"] = Field(
        ..., description="Final recommendation: 'continue_cleaning' or 'stop_cleaning'."
    )
    flags: List[str] = Field(
        default_factory=list,
        description="Diagnostic flags from all tiers (e.g., 'inf_nan_present: col1, col2')."
    )


# Schema definition for the entire dataset
class DatasetProfile(BaseModel):
    row_count: int = Field(..., description="Total number of rows in the raw dataset.")
    columns: List[ColumnProfile] = Field(..., description="List of profiles for each column.")
    algorithmic_quality_score: Optional[AlgorithmicQualityScore] = Field(
        None, description="Deterministic quality score computed from profile statistics. "
                          "This is the authoritative quality metric — LLM scores are advisory only."
    )


################################################################################
# Schema Definitions for Investigation Findings:
# The structured output of the Investigator Agent. This is the "handoff 
# document" between the Investigator and the Code Generator. It captures
# WHAT is wrong with the data, without prescribing HOW to fix it in code.
#
# This separation means:
#   - Findings persist across code-generation retries (no re-investigation)
#   - The answer agent can read findings to caveat its response
#   - We get a clean audit trail of what the LLM detected
################################################################################

class SemanticViolation(BaseModel):
    """A single data quality issue discovered by the Investigator."""
    violation_id: int = Field(..., description="Unique ID for this violation (1-indexed).")
    severity: Literal["CRITICAL", "INFO"] = Field(
        ..., description="CRITICAL = must be fixed before modeling. INFO = worth noting but does not block modeling."
    )
    category: Literal[
        "SENTINEL_VALUE",       # -1, 999, "N/A" masquerading as real data
        "TEMPORAL_VIOLATION",   # Events out of causal order
        "CROSS_COLUMN_LOGIC",   # Impossible combinations across columns
        "TYPE_ERROR",           # Column dtype doesn't match semantic meaning
        "MISSING_DATA",         # Nulls/gaps in TARGET column only (feature missingness handled by AutoGluon)
        "OUTLIER",              # Statistical outliers or impossible values
        "DUPLICATE",            # Duplicate rows or near-duplicates
        "PII_DETECTED",         # Personally identifiable information
        "FORMAT_INCONSISTENCY", # Mixed formats within a column
        "OTHER"
    ] = Field(..., description="Category of the violation.")
    
    affected_columns: List[str] = Field(
        ..., description="Which column(s) are involved."
    )
    description: str = Field(
        ..., description="Human-readable explanation of what's wrong. "
                         "E.g., 'Column age contains 847 rows with value -1, likely a sentinel for missing data.'"
    )
    evidence: str = Field(
        ..., description="Specific evidence from the profile or tool calls that supports this finding. "
                         "E.g., 'top_frequent_values shows -1 with count=847 (8.5% of rows). "
                         "min_value=-1, but age cannot be negative.'"
    )
    suggested_action: str = Field(
        ..., description="Plain-English suggestion for how to fix this. NOT code — just intent. "
                         "E.g., 'Replace -1 values in age with NaN.'"
    )


class ColumnDropRationale(BaseModel):
    """A single column-drop decision with its reason."""
    column: str = Field(..., description="Column name being dropped.")
    reason: str = Field(..., description="Why this column is being dropped.")


class InvestigationFindings(BaseModel):
    """Complete output of the Investigator Agent."""

    target_column: Optional[str] = Field(
        None, description="The column that best answers the user's query (prediction target). "
                          "None if the investigator couldn't determine it."
    )
    target_column_rationale: Optional[str] = Field(
        None, description="Why this column was chosen as the target."
    )
    task_type: Optional[Literal[
        "binary_classification", "multiclass_classification", "regression"
    ]] = Field(None, description="The ML task type implied by the target column.")

    violations: List[SemanticViolation] = Field(
        default_factory=list, description="All semantic violations found in the data."
    )

    columns_to_drop: List[str] = Field(
        default_factory=list,
        description="Columns that should be dropped entirely (e.g., PII, IDs, constant columns)."
    )
    columns_to_drop_rationale: List[ColumnDropRationale] = Field(
        default_factory=list,
        description="List of columns being dropped and their reasons."
    )
    
    key_caveats: List[str] = Field(
        default_factory=list,
        description="Important caveats the answer agent should mention when presenting results. "
                    "E.g., 'Income column was 40% sentinel values — predictions involving income may be unreliable.'"
    )



################################################################################
# Schema Definitions for Agent to Cleaner:
# These schemas define the exact structure of the JSON files exchanged between
# the Code Generator Agent and the cleaning sandbox.
################################################################################

class CleaningStep(BaseModel):
    step_id: int = Field(..., description="The execution order (1-indexed).")
    operation: OperationType = Field(..., description="The category of operation being performed.")
    target_column: Optional[str] = Field(None, description="The specific column being modified (if applicable).")

    @field_validator("operation", mode="before")
    @classmethod
    def normalize_operation(cls, v):
        if isinstance(v, str):
            v = v.strip().upper()
            aliases = {
                "DROP_COLUMNS": "DROP_COLUMN",
                "RENAME_COLUMNS": "RENAME_COLUMN",
                "DROP_ROW": "DROP_ROWS",
                "CAST_TYPES": "CAST_TYPE",
                "CUSTOM": "CUSTOM_CODE",
            }
            v = aliases.get(v, v)
        return v

    @field_validator("target_column", mode="before")
    @classmethod
    def coerce_target_column(cls, v):
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return v

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the operation."
    )
    
    # Links back to the investigation finding that motivated this step
    addresses_violation: Optional[int] = Field(
        None, description="The violation_id from InvestigationFindings that this step addresses. "
                          "Provides traceability from code back to reasoning."
    )
    
    justification: str = Field(..., description="Semantic reasoning: WHY are we doing this? (e.g., 'Detected -1 in Age column, likely sentinel').")
    
    python_code: str = Field(
        ..., 
        description="The actual executable Python code snippet using 'df'. E.g., `df = df[df['age'] > 0]`"
    )

class CleaningRecipe(BaseModel):
    """The full plan returned by the Code Generator Agent."""
    steps: List[CleaningStep]


################################################################################
# Schema Definitions for Audit Trail:
# Tracks the actual history of what was executed in the Sandbox.
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
# Input schemas for investigation tools. Each tool function uses one of these
# as its args_schema so LangChain can generate the correct tool-call JSON.
################################################################################

class InspectRowsInput(BaseModel):
    query: str = Field(..., description="Pandas query string to filter rows (e.g., 'age < 0').")
    limit: int = Field(5, description="Number of rows to return (default 5). Keep this small to save tokens.")

class CrossColumnFrequencyInput(BaseModel):
    col_a: str = Field(..., description="First column name for the crosstab.")
    col_b: str = Field(..., description="Second column name for the crosstab.")
    top_n: int = Field(10, description="Max number of combination pairs to return.")

class TemporalOrderingCheckInput(BaseModel):
    date_col_a: str = Field(..., description="The column expected to occur FIRST chronologically.")
    date_col_b: str = Field(..., description="The column expected to occur SECOND chronologically.")
    sample_violations: int = Field(5, description="Number of violating rows to sample.")

class ValueDistributionInput(BaseModel):
    column: str = Field(..., description="Column name to compute distribution for.")
    bins: int = Field(20, description="Number of histogram bins for numeric columns.")

class NullCoOccurrenceInput(BaseModel):
    threshold: float = Field(0.5, description="Minimum co-occurrence ratio to report (0.0-1.0).")

class CorrelationScanInput(BaseModel):
    target_column: str = Field(..., description="Column to compute correlations against.")
    top_n: int = Field(10, description="Number of top correlated columns to return.")

class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query to verify a data quality hypothesis. Be specific — include units, ranges, or domain context.")
    domains: Optional[List[str]] = Field(None, description="Optional list of domains to restrict results to authoritative sources (e.g., ['who.int', 'mayoclinic.org'] for medical data, ['sec.gov', 'reuters.com'] for financial data).")