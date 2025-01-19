import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

import sys, os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import utils


tab_gaussian, tab_Cornish = st.tabs(["Gaussian distribution", "Cornish-Fisher expansion"])

with tab_gaussian:
    st.markdown("We suppose that the returns of an asset follow a Gaussian distribution. This distribution "
                "remains the same for every period but are independent")
    # Gaussian distribution
    volatility = st.number_input("Volatility", min_value=0.0, max_value=1000000000.0, value=1.0)
    mean = st.number_input("Mean", min_value=-100.0, max_value=1000000000.0, value=0.0)
    time = st.number_input("Time in units of time", min_value=0, max_value=100, value=1)

    mean = mean * time
    volatility = volatility * np.sqrt(time)

    x = np.linspace(mean-3*volatility, mean+3*volatility, 100000)

    y = stats.norm.pdf(x, mean, volatility)
    cdf = stats.norm.cdf(x, mean, volatility)

    # find the x% quantile
    var_conf = st.number_input("VaR & ES confidence level", min_value=0.0, max_value=1.0, value=0.95)
    # es_conf = st.number_input("ES confidence level", min_value=0.0, max_value=1.0, value=0.95)

    # Find the VaR and ES
    index_x_geq_conf = np.where(cdf <= 1-var_conf)[0]
    x_conf = x[index_x_geq_conf[-1]]
    prob_conf = cdf[index_x_geq_conf[-1]]

    VaR, ES = utils.calculate_var_es(mean, volatility**2, var_conf)

    # Create the plot
    fig, ax = plt.subplots()
    # Plot line and area under the curve
    ax.plot(x, y, color="white", label="Line Chart")  # Line chart with white color
    ax.fill_between(x[index_x_geq_conf], y[index_x_geq_conf], alpha=0.3, color="red",
                    label=f"VaR = {np.round(VaR, 2)}\n"
                          f"ES = {np.round(ES, 2)}")

    # Set tick params to white
    ax.tick_params(colors="white")

    # Set spines to white
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    # Set the background transparent
    fig.patch.set_alpha(0)  # Transparent figure background
    ax.set_facecolor("none")  # Transparent axes background

    # Set legend
    ax.legend(facecolor="none", edgecolor="white", labelcolor="white")

    # Display in Streamlit
    st.pyplot(fig)

    # Create the plot
    fig, ax = plt.subplots()
    # Plot line and area under the curve
    ax.plot(x, cdf, color="white", label="Line Chart")  # Line chart with white color
    ax.fill_between(x[index_x_geq_conf], cdf[index_x_geq_conf], alpha=0.3, color="red",
                    label=f"VaR = {np.round(VaR, 2)}\n"
                          f"ES = {np.round(ES, 2)}")

    # Set tick params to white
    ax.tick_params(colors="white")

    # Set spines to white
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    # Set the background transparent
    fig.patch.set_alpha(0)  # Transparent figure background
    ax.set_facecolor("none")  # Transparent axes background

    # Set legend
    ax.legend(facecolor="none", edgecolor="white", labelcolor="white")

    # Display in Streamlit
    st.pyplot(fig)

with tab_Cornish:
    vol = st.number_input("Volatility", min_value=0.0, max_value=1000000000.0, value=1.0, key="vol")
    mean = st.number_input("Mean", min_value=-100.0, max_value=1000000000.0, value=0.0, key="mean")
    conf = st.number_input("VaR & ES confidence level", min_value=0.0, max_value=1.0, value=0.95, key="conf")
    skew = st.number_input("Skewness", min_value=-100.0, max_value=1000000000.0, value=0.0, key="skew")
    kur = st.number_input("Kurtosis", min_value=-100.0, max_value=1000000000.0, value=3.0, key="kur")

    st.latex(r"z_{\alpha}^{CF} \approx z_{\alpha} + \frac{(z_{\alpha}^2 - 1) S}{6} +"
             r" \frac{(z_{\alpha}^3 - 3z_{\alpha}) (K-3)}{24} - \frac{(2z_{\alpha}^3 - 5z_{\alpha}) S^2}{36}")

    st.write(f"VAR: {utils.cornish_fisher_var_es(mean, vol**2, conf, skew, kur)[0]}")