import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import ast
import networkx as nx

import sys, os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import utils


tab_from_array, tab_binomial, tab_gaussian, tab_Cornish = st.tabs(["Returns array",
                                                                     "Binomial distribution",
                                                                     "Gaussian distribution",
                                                                     "Cornish-Fisher expansion"])

with tab_from_array:

    st.text('We compute the VAR and ES of an asset given its returns array. The array can be unsorted.')
    # Input array as a string
    array_str = st.text_input("Enter an array (e.g., [1, 2, 3])")
    array = [1, 2, 3]

    if array_str:
        try:
            # Safely evaluate the input string as a Python literal
            array = ast.literal_eval(array_str)
            if isinstance(array, list):
                st.write("You entered the array:", array)
            else:
                st.error("Input is not a valid array!")
        except Exception as e:
            st.error(f"Invalid input: {e}")

    # Sort the array
    array = np.array([array])
    array = np.sort(array)
    array = array.squeeze(0)

    # Find the x% quantile
    var_conf = st.number_input("VaR & ES confidence level", min_value=0.0, max_value=1.0, value=0.95, key="var_array")
    # es_conf = st.number_input("ES confidence level", min_value=0.0, max_value=1.0, value=0.95)

    # Get only the returns that are below the x% quantile
    index_x_geq_conf = int(array.shape[0] * (1-var_conf))
    x_conf = array[index_x_geq_conf]
    x_less = array[:index_x_geq_conf+1]
    # VaR is the smallest value in the array that is greater than the x% quantile
    VaR = x_conf
    # ES is the average of the returns that are below the x% quantile
    ES = np.mean(x_less)

    st.latex(f"VaR({var_conf}) = {VaR}")
    st.latex(f"ES({var_conf}) = {ES}")

with tab_binomial:

    st.text('We compute the VAR and ES of 2 assets given their binomial distributions.'
            'If using only one asset, set the gain and loss of the second one to 0.')
    p_1Loss = st.number_input("Asset 1 probability of loss", min_value=0.0, max_value=1.0, value=0.5)
    Loss1 = st.number_input("Asset 1 Loss (Losses require a [-] sign ! )", min_value=-1000000000.0, max_value=1000000000.0, value=1.0)
    Gain1 = st.number_input("Asset 1 Gain (Losses require a [-] sign ! )", min_value=-1000000000.0, max_value=1000000000.0, value=1.0)
    p_2Loss = st.number_input("Asset 2 probability of loss", min_value=0.0, max_value=1.0, value=0.5)
    Loss2 = st.number_input("Asset 2 Loss (Losses require a [-] sign ! )", min_value=-1000000000.0, max_value=1000000000.0, value=1.0)
    Gain2 = st.number_input("Asset 2 Gain (Losses require a [-] sign ! )", min_value=-1000000000.0, max_value=1000000000.0, value=1.0)
    Confidence = st.number_input("Confidence level", min_value=0.0, max_value=1.0, value=0.95, key="conf_binomial")
    # Calculate probabilities
    p_L1 = p_1Loss
    p_W1 = 1 - p_L1
    p_L2 = p_2Loss
    p_W2 = 1 - p_L2

    # Calculate the returns of the assets in case of loss and gain
    R_1L2L = Loss1 + Loss2
    R_1L2G = Loss1 + Gain2
    R_1G2L = Gain1 + Loss2
    R_1G2G = Gain1 + Gain2

    returns_and_probs = {
        "1L2L": {
            "Return": R_1L2L,
            "Probability": p_L1*p_L2
        },
        "1L2G": {
            "Return": R_1L2G,
            "Probability": p_L1*p_W2
        },
        "1G2L": {
            "Return": R_1G2L,
            "Probability": p_W1*p_L2
        },
        "1G2G": {
            "Return": R_1G2G,
            "Probability": p_W1*p_W2
        }
    }

    df = pd.DataFrame(returns_and_probs).T
    # Merge the row with the same return
    df = df.groupby("Return").sum()
    # Sort the dataframe by the returns, the greatest is the first
    df = df.sort_values(by="Return", ascending=False)
    df["Cumulative Probability"] = df["Probability"].cumsum()
    st.write(df)

    # Compute the VaR
    try:
        VaR = df[df["Cumulative Probability"] >= Confidence]
        VaR = VaR.reset_index()
        VaR = VaR["Return"].iloc[0]

        df_conf = df[df["Cumulative Probability"] >= Confidence]
        df_conf = df_conf.reset_index()
        cumsum_less_first = np.sum(df_conf["Probability"].iloc[1:])
        prob_first = 1 - Confidence - cumsum_less_first
        df_conf["Probability"].iloc[0] = prob_first
        ES = df_conf["Return"].T @ df_conf["Probability"] / (1 - Confidence)

    except Exception as e:
        st.error(f"df is empty")
        st.error(f"Error: {e}")

    st.latex(f"VaR({Confidence}) = {VaR}")
    st.latex(f"ES({Confidence}) = {ES}")


with tab_gaussian:
    st.markdown("We suppose that the returns of an asset follow a Gaussian distribution. This distribution "
                "remains the same for every period but are independent")
    st.markdown("VaR_sqrt is the result with the teacher's formula, VaR my results by modifying the original distribution.")
    # Gaussian distribution
    volatility = st.number_input("Volatility", min_value=0.0, max_value=1000000000.0, value=1.0)
    mean_ = st.number_input("Mean", min_value=-100.0, max_value=1000000000.0, value=0.0)
    time = st.number_input("Time in units of time", min_value=0, max_value=100, value=1)

    mean = mean_ * time
    mean_sqrt = mean_ * np.sqrt(time)
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
    Var_sqrt, ES_sqrt = utils.calculate_var_es(mean_sqrt, volatility**2, var_conf)

    # Create the plot
    fig, ax = plt.subplots()
    # Plot line and area under the curve
    ax.plot(x, y, color="white", label="Line Chart")  # Line chart with white color
    ax.fill_between(x[index_x_geq_conf], y[index_x_geq_conf], alpha=0.3, color="red",
                    label=f"VaR = {np.round(VaR, 2)}\n"
                          f"VaR sqrt = {np.round(Var_sqrt, 2)}\n"
                          f"ES = {np.round(ES, 2)}\n"
                          f"ES sqrt = {np.round(ES_sqrt, 2)}")

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
                          f"VaR sqrt = {np.round(Var_sqrt, 2)}\n"
                          f"ES = {np.round(ES, 2)}\n"
                          f"ES sqrt = {np.round(ES_sqrt, 2)}")

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

    st.latex(r"VaR = \mu + z_{\alpha}^{CF} \sigma")

    st.latex(rf"VaR({conf}): {utils.cornish_fisher_var_es(mean, vol**2, conf, skew, kur)[0]}")
