################################################################################
# AutoGluon model training module.
#
# STUB: Not yet implemented. Returns placeholder metadata so the LangGraph
# pipeline can run end-to-end without training a real model.
#
# When ready, replace this with the actual AutoGluon training logic.
################################################################################

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def train_model(
    df: pd.DataFrame,
    user_query: str,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train an ML model on the cleaned data.

    STUB — returns placeholder metadata. Replace with AutoGluon when ready.
    """
    print("    [STUB] AutoGluon training skipped.")
    return {
        "stub": True,
        "target_column": target_column,
        "message": "AutoGluon training not yet implemented.",
    }
