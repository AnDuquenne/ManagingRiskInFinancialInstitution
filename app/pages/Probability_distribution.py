import streamlit as st

import scipy.stats as stats
import numpy as np
import pandas as pd

import sys
import os
import io

import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

gaussian, student_t = st.tabs(["Gaussian distribution", "Student t distribution"])

with gaussian:
    st.header("Gaussian distribution")
    # Gaussian distribution
    x = np.linspace(-5, 5, 1000)
    volatility = st.number_input("Volatility", min_value=0.0, max_value=100.0, value=1.0)
    mean = st.number_input("Mean", min_value=-100.0, max_value=100.0, value=0.0)
    y = stats.norm.pdf(x, mean, volatility)
    chart_data = pd.DataFrame({
        "x": x,
        "y": y
    })
    st.line_chart(chart_data, x="x", y="y")

with student_t:
    # Student t distribution
    v = st.number_input("Degrees of freedom", min_value=1, max_value=1000, value=5)
    normal = stats.norm.pdf
    x = np.linspace(-5, 5, 1000)
    t = stats.t.pdf(x, v)
    y = normal(x)
    chart_data = pd.DataFrame({
        "x": x,
        "normal": y,
        "t": t
    })
    st.line_chart(chart_data, x="x", y=["normal", "t"])


