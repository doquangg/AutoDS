################################################################################
# This file documents the exact layout of the agent, dictating the exact
# information it contains.
################################################################################
import operator
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
import pandas as pd
from langchain_core.messages import BaseMessage
from core.schema import DatasetProfile, CleaningRecipe, CleaningLogEntry

class AgentState(TypedDict):
    # --- Inputs ---
    user_query: str  
    
    # --- Data Assets ---
    working_df: pd.DataFrame 
    clean_df: Optional[pd.DataFrame]
    model_metadata: Optional[str] 
    
    # --- Conversation & Context ---
    messages: Annotated[List[BaseMessage], operator.add]
    
    # --- Cleaning Artifacts ---
    profile: Optional[DatasetProfile]
    current_plan: Optional[CleaningRecipe]
    
    # --- Audit Trail ---
    cleaning_history: Annotated[List[CleaningLogEntry], operator.add]
    
    # --- Control Flow ---
    retry_count: int
    latest_error: Optional[str]
    final_answer: Optional[str]