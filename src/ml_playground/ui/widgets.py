"""Small Streamlit adapters for declarative hyperparameter widgets."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ml_playground.models.specifications import UNSET


def parameter_widget(parameter: Any, *, key_prefix: str) -> Any:
    """Render a widget from the shared HyperparameterSpec metadata."""
    label = parameter.name.replace("_", " ").capitalize()
    help_text = parameter.description or None
    if parameter.choices is not None:
        default = parameter.default if parameter.default is not UNSET else parameter.choices[0]
        return st.selectbox(
            label,
            options=parameter.choices,
            index=parameter.choices.index(default),
            help=help_text,
            key=f"{key_prefix}_{parameter.name}",
            format_func=lambda value: "None (default)" if value is None else str(value),
        )
    value_types = parameter.value_type if isinstance(parameter.value_type, tuple) else (parameter.value_type,)
    if parameter.optional and int in value_types and type(None) in value_types:
        has_limit = st.checkbox(
            f"Set a maximum for {label.lower()}",
            value=parameter.default is not None,
            help=help_text,
            key=f"{key_prefix}_{parameter.name}_enabled",
        )
        if not has_limit:
            return None
        minimum = int(parameter.minimum) if parameter.minimum is not None else 1
        maximum = int(parameter.maximum) if parameter.maximum is not None else 100
        default = parameter.default if parameter.default is not None else minimum
        return st.number_input(
            label, min_value=minimum, max_value=maximum, value=int(default),
            step=int(parameter.step or 1), help=help_text, key=f"{key_prefix}_{parameter.name}",
        )
    if parameter.value_type is int:
        minimum = int(parameter.minimum) if parameter.minimum is not None else None
        maximum = int(parameter.maximum) if parameter.maximum is not None else None
        default = parameter.default if parameter.default is not UNSET else minimum or 0
        return st.number_input(
            label, min_value=minimum, max_value=maximum, value=int(default),
            step=int(parameter.step or 1), help=help_text, key=f"{key_prefix}_{parameter.name}",
        )
    if parameter.value_type is float:
        minimum = float(parameter.minimum) if parameter.minimum is not None else None
        maximum = float(parameter.maximum) if parameter.maximum is not None else None
        default = parameter.default if parameter.default is not UNSET else minimum or 0.0
        return st.number_input(
            label, min_value=minimum, max_value=maximum, value=float(default),
            step=float(parameter.step or 0.01), help=help_text, key=f"{key_prefix}_{parameter.name}",
        )
    if parameter.value_type is bool:
        return st.checkbox(
            label, value=parameter.default, help=help_text, key=f"{key_prefix}_{parameter.name}"
        )
    raise TypeError(f"No Streamlit widget mapping for parameter '{parameter.name}'.")
