import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

tab_Credit_loss_computation, _ = st.tabs(["Credit loss computation", "next tab"])

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
