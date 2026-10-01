"""Pandera schema for the feature vector inference-api accepts.

Column set and order must match ml.feature_extraction.FEATURE_NAMES —
that module is the single source of truth for the feature contract.
Built dynamically (rather than a static DataFrameModel) since the
column list is generated, not fixed at class-definition time.
"""

import pandera as pa

from feature_extraction import FEATURE_NAMES

FEATURE_SCHEMA = pa.DataFrameSchema(
    {name: pa.Column(float, nullable=False, coerce=True) for name in FEATURE_NAMES},
    strict=True,
)
