import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import ast
import math

tab_Credit_loss_computation, tab_merton, tab_altman, tab_def_prob = st.tabs(["Credit loss computation",
                                                               "Merton's model",
                                                               "Altman z-score",
                                                               "Default Probability"])

with tab_Credit_loss_computation:
    with st.expander("Credit loss computation of 2 independent assets"):
        st.header("Expected loss computation")
        st.write("We consider a portfolio of two instruments with different default risk and returns. The two instruments are independent.")

        # Instrument 1
        st.subheader("Instrument 1")
        value_1 = st.number_input("Value of the instrument 1", min_value=0.0, max_value=10000000000.0, value=100.0)
        pod_1 = st.number_input("Probability of default of the instrument 1", min_value=0.0, max_value=1.0, value=0.01)
        recovery_rate_1 = st.number_input("Recovery rate of the instrument 1", min_value=0.0, max_value=1.0, value=0.5)

        # Instrument 2
        st.subheader("Instrument 2")
        value_2 = st.number_input("Value of the instrument 2", min_value=0.0, max_value=10000000000.0, value=100.0)
        pod_2 = st.number_input("Probability of default of the instrument 2", min_value=0.0, max_value=1.0, value=0.05)
        recovery_rate_2 = st.number_input("Recovery rate of the instrument 2", min_value=0.0, max_value=1.0, value=0.5)

        # Expected loss computation
        st.subheader("Expected loss computation")
        expected_loss_1 = value_1 * pod_1 * (1 - recovery_rate_1)
        expected_loss_2 = value_2 * pod_2 * (1 - recovery_rate_2)
        total_expected_loss = expected_loss_1 + expected_loss_2
        st.write(f"Expected loss of instrument 1: {expected_loss_1:.2f}")
        st.write(f"Expected loss of instrument 2: {expected_loss_2:.2f}")
        st.write(f"Total expected loss: {total_expected_loss:.2f}")

    with st.expander("Survival rate of a company"):
        st.header("Survival rate of a company")
        nb_years = st.number_input("Number of years", min_value=1, max_value=100, value=3)
        default_probs = []
        for i in range(nb_years):
            st.write(f"Year {i+1}")
            default_probs.append(st.number_input(f"Default probability of the company at year {i+1}", min_value=0.0, max_value=1.0, value=0.9))

        # Compute the probability of survival
        survival_rate = np.prod([1 - p for p in default_probs])
        st.write(f"Survival rate of the company: {survival_rate*100:.2f}%")

with tab_merton:
    with st.expander("Probability of default"):
        st.write("Under merton's model (assume that asset returns are lognormally distributed), it can be shown that the probability of default is:")
        st.latex(r"PD = P(A_T < N \mid A_t) = N(-d_2) = 1 - N(d_2)")
        st.write("Where:")
        st.latex(r"d_1 = \frac{\ln\left(\frac{A(t)}{N}\right) + \left(r_f + \frac{\sigma_A^2}{2}\right) \cdot (T - t)}{\sigma_A \cdot \sqrt{T - t}},")
        st.latex(r"d_2 = d_1 - \sigma_A \cdot \sqrt{T - t},")
        st.write("and:")
        st.write("A(t) is the asset value at time t, N is the face value of the debt (bond)"
                 " or strike price, r_f is the risk-free rate,")
        st.write(r"sigma(A) is the asset volatility, T is the time to maturity, and t is the current time.")

        st.write("Capital structure initially (t=0):")
        bond_face_value = st.number_input("Face value of the bond", min_value=0.0, max_value=10000000000.0, value=100.0)
        bond_face_date = st.number_input("Face date of the bond (in years)", min_value=0, max_value=100, value=1)
        asset_value = st.number_input("Value of the firm's asset", min_value=0.0, max_value=10000000000.0, value=100.0)
        rf = st.number_input("Risk-free rate", min_value=0.0, max_value=1.0, value=0.05)
        asset_volatility = st.number_input("Annual asset volatility", min_value=0.0, max_value=1.0, value=0.2)

        d1 = (np.log(asset_value / bond_face_value) + (rf + 0.5 * asset_volatility**2) * bond_face_date) / (asset_volatility * np.sqrt(bond_face_date))
        d2 = d1 - asset_volatility * np.sqrt(bond_face_date)
        pd = 1 - stats.norm.cdf(d2)

        st.latex(f"d_1 = {d1:.8f}")
        st.latex(f"d_2 = {d2:.8f}")
        st.latex(f"PD = {pd:.8f}")

    with st.expander("Distance to default"):
        st.write("The distance to default is defined as (t=0):")
        st.latex(r"DD_t = \frac{\ln(A_t) - \ln(N) + \left(r_f - \frac{\sigma_A^2}{2}\right) \cdot (T - t)}{\sigma_A \cdot \sqrt{T - t}}")
        st.write("Capital structure initially:")
        bond_face_value = st.number_input("Face value of the bond", min_value=0.0, max_value=10000000000.0, value=100.0, key="bond_face_value_dd")
        bond_face_date = st.number_input("Face date of the bond (in years)", min_value=0, max_value=100, value=1, key="bond_face_date_dd")
        asset_value = st.number_input("Value of the firm's asset", min_value=0.0, max_value=10000000000.0, value=100.0, key="asset_value_dd")
        rf = st.number_input("Risk-free rate", min_value=0.0, max_value=1.0, value=0.05, key="rf_dd")
        asset_volatility = st.number_input("Annual asset volatility", min_value=0.0, max_value=1.0, value=0.2, key="asset_volatility_dd")
        DD = (np.log(asset_value) - np.log(bond_face_value) + (rf - 0.5 * asset_volatility**2) * bond_face_date) / (asset_volatility * np.sqrt(bond_face_date))
        st.latex(f"DD = {DD:.8f}")

with tab_altman:
    # Explanation
    st.markdown(r"""
    Altman (1968)'s Z-score is a financial model used to predict the likelihood of bankruptcy for a company.
    The formula is:

    $$
    Z = 1.2X_1 + 1.4X_2 + 3.3X_3 + 0.6X_4 + 0.999X_5
    $$

    Where:
    $$
    \begin{align*}
    X_1 &= \frac{\text{Working Capital}}{\text{Total Assets}}\\
    X_2 &= \frac{\text{Retained Earnings}}{\text{Total Assets}}\\
    X_3 &= \frac{\text{EBIT}}{\text{Total Assets}}\\
    X_4 &= \frac{\text{Market Value of Equity}}{\text{Total Liabilities}}\\
    X_5 &= \frac{\text{Sales}}{\text{Total Assets}}\\
    \end{align*}
    $$
    """)

    wc_ = st.number_input("Working Capital", min_value=0.0, max_value=10000000000.0, value=100.0)
    re_ = st.number_input("Retained Earnings", min_value=0.0, max_value=10000000000.0, value=100.0)
    ebit_ = st.number_input("EBIT", min_value=0.0, max_value=10000000000.0, value=100.0)
    mve_ = st.number_input("Market Value of Equity", min_value=0.0, max_value=10000000000.0, value=100.0)
    sales_ = st.number_input("Sales", min_value=0.0, max_value=10000000000.0, value=100.0)
    total_lia_ = st.number_input("Total Liabilities", min_value=0.0, max_value=10000000000.0, value=100.0)
    total_assets_ = st.number_input("Total Assets", min_value=0.0, max_value=10000000000.0, value=100.0)

    X1 = wc_ / total_assets_
    X2 = re_ / total_assets_
    X3 = ebit_ / total_assets_
    X4 = mve_ / total_lia_
    X5 = sales_ / total_assets_

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 0.999*X5

    st.latex(f"Z = {Z:.8f}")

    st.latex(r'''Z > 3: \text{Company is unlikely to default}''')
    st.latex(r'''2.7 \geq Z \leq 3.0: \text{Company is on alert}''')
    st.latex(r'''1.8 \geq Z \leq 2.7: \text{Company has a certain risk to default}''')
    st.latex(r'''Z < 1.8: \text{Company defaults with a very high likelihood}''')

with tab_def_prob:
    with st.expander("Default probability from cumulative probability of default"):
        # Input array as a string
        array_str = st.text_input("Enter the cumulative probability of default as an array (e.g., [1, 2, 3])")
        array = [0.1, 0.2, 0.3]

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
        array = np.array(array)

        print(array)

        dt = st.number_input("Default time", min_value=0, max_value=100, value=1)
        upto = st.number_input("Conditional up to time: ", min_value=0, max_value=100, value=1)

        # Bond unconditional default probability
        st.write("Bond unconditional default probability:")
        st.latex(rf"P(D)_t = {array[dt-1]-array[dt-2]:.8f}")
        st.write("Bond conditional default probability:")
        st.latex(rf"P(D|T)_t = {(array[dt-1]-array[upto-1])/(1 - array[upto-1]):.8f}")

    with st.expander("Probability of several defaults"):
        nb_def = st.number_input("Number of defaults", min_value=0, max_value=100, value=1)
        p_def = st.number_input("Probability of default", min_value=0.0, max_value=1.0, value=0.01)
        nb_loans = st.number_input("Number of loans (or other instruments)", min_value=0, max_value=100, value=100)

        # Poisson law
        poisson = stats.poisson.pmf(nb_def, nb_loans * p_def)
        poisson_2 = (math.factorial(nb_loans) / (math.factorial(nb_def) * math.factorial(nb_loans - nb_def))) * (p_def ** nb_def) * ((1 - p_def) ** (nb_loans - nb_def))

        st.latex(f"P(X = {nb_def}) = {poisson_2:.8f}")


