import streamlit as st
import streamlit.components.v1 as components
import os

_component = components.declare_component(
    "geolocation_component",
    path=os.path.join(os.path.dirname(__file__), "frontend.html")
)

def get_geolocation(key: str = None):
    return _component(key=key, default=None)
