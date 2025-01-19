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

gaussian, student_t, varcov = st.tabs(["Gaussian distribution", "Student t distribution",
                                       "Variance-covariance computation"])

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


with varcov:
    with st.expander("Variance of 2 Standard Normal random variables, given the covariance"):
        st.text("We suppose that X and Y are two standard normal random variables with covariance rho.")
        st.latex(r"X, Y \sim N(0, 1)")
        st.text("We compute the variance of:")
        st.latex(r"\alpha X + \beta Y")
        alpha = st.number_input("Alpha", min_value=-100.0, max_value=100.0, value=1.0)
        beta = st.number_input("Beta", min_value=-100.0, max_value=100.0, value=1.0)
        rho = st.number_input("Covariance", min_value=-1.0, max_value=1.0, value=0.0)

        st.text("The variance of the linear combination is:")
        st.latex(r"\alpha^2 \operatorname{Var}(X) + \beta^2 \operatorname{Var}(Y) + 2 \alpha \beta \operatorname{Cov}(X, Y)")
        st.text("Where Var(X) = Var(Y) = 1")
        variance = alpha**2 + beta**2 + 2 * alpha * beta * rho
        st.latex(f"VAR({alpha}X + {beta}B)= {variance:.8f}")
