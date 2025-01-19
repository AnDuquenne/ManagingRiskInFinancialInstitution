import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

st.header("Volatility Modelling")

tab_EWMA, tab_Garch = st.tabs(["EWMA", "Garch(1, 1) model"])

with tab_EWMA:
    st.html(r'<h2>Exponentially Weighted Moving Average</h2>')

    with st.expander("Compute decay factor"):

        df = st.number_input("Weight decay factor", min_value=0.0, max_value=1.0, value=0.9)
        nb_samples = st.number_input("Number of samples", min_value=1, max_value=1000, value=25)
        x = np.arange(1, nb_samples)
        # Weights in the EWMA
        decay_factor = np.array([
            df ** (i+1) for i in range(x.shape[0])
        ])
        weights = df * ((1 - df) / (df*(1-df**nb_samples)))
        st.line_chart(pd.DataFrame({"x": x,
                                    "decay_factor": decay_factor * weights}),
                      x="x", y=["decay_factor"])

        st.markdown("If we make the assumption that the periods tends to infinity, we can compute the weight as follows:")
        last_day_weight = 1 - df
        last_days_weights = [last_day_weight * (df ** i) for i in range(1, x.shape[0])]
        lf = [last_day_weight] + last_days_weights
        st.line_chart(pd.DataFrame({"x": x,
                                    "weight": lf}),
                      x="x", y="weight")

    with st.expander("Compute volatility"):
        st.write("The volatility can be computed using the following formula:")
        st.latex(r'\hat{\sigma}_t^2=\frac{1-\lambda}{\lambda\left(1-\lambda^p\right)} \sum_{i=1}^p \lambda^i \cdot R_{t-i}^2')
        st.write("Where:")
        st.write(r"$\lambda$ is the decay factor")
        st.write(r"$R^2_{t-1}$ is the squared return at time $t-1$")
        st.write(r"$\sigma^2_{t-1}$ is the volatility at time $t-1$")
        st.write(r"$\sigma_t$ is the volatility at time $t$")
        st.html(r'<h3>Update the volatility</h3>')

        sigma_t_1 = st.number_input("Volatility at time t-1", min_value=-100.0, max_value=100.0, value=1.0)
        R_t_1 = st.number_input("Return at time t-1", min_value=-100.0, max_value=100.0, value=1.0)
        df_ = st.number_input("Decay factor", min_value=0.0, max_value=1.0, value=0.94)

        st.write(r"The volatility at time t can be computed using $p \rightarrow \infty$")
        st.latex(r'\hat{\sigma}_t^2=(1-\lambda) R_{t-1}^2+\lambda \sigma_{t-1}^2')

        sigma_t = np.sqrt((1 - df_) * R_t_1**2 + df_ * sigma_t_1**2)

        st.latex(rf'\sigma_t = \sqrt{{(1 - \lambda) R^2_{{t-1}} + \lambda \sigma^2_{{t-1}}}}')
        st.latex(rf'\sigma_t = {sigma_t:.7f}')
        
        # Evolution of the sigma depending on the decay factor
        x = np.arange(50, 100, 1)/100
        sigma = np.sqrt((1 - x) * R_t_1**2 + x * sigma_t_1**2)
        st.line_chart(pd.DataFrame({"lambda": x,
                                    "sigma": sigma}),
                      x="lambda", y="sigma")

with tab_Garch:
    with st.expander("Single-period volatility forecast"):
        # Garch model
        w = st.number_input("w", min_value=0.0, max_value=100.0, value=1.0)
        alpha = st.number_input("alpha", min_value=0.0, max_value=1.0, value=0.1)
        beta = st.number_input("beta", min_value=0.0, max_value=1.0, value=0.1)

        st.write("The Garch(1, 1) model is defined as follows:")
        t1 = r"{t-1}"
        st.latex(rf'\sigma^2_t = {w} + {alpha} R^2_{t1} + {beta} \sigma^2_{t1}')

        st.write(r"The unconditional variance of a GARCH(1, 1) process can be computed using:")
        st1 = r'\sigma^2 = \frac{w}{1 - \alpha - \beta}'
        sigma_t_squared = w / (1 - alpha - beta)
        st2 = rf' = {sigma_t_squared}'
        st.latex(st1 + st2)
        st.latex(rf'\sigma = {sigma_t_squared**(1/2)*100:.2f}\%')

    with st.expander("Multi-period volatility forecast"):
        w_multi = st.number_input("w", min_value=0.0, max_value=100.0, value=1.0, key="w_multi")
        alpha_multi = st.number_input("alpha", min_value=0.0, max_value=1.0, value=0.1, key="alpha_multi")
        beta_multi = st.number_input("beta", min_value=0.0, max_value=1.0, value=0.1, key="beta_multi")
        sigma_t_squared_multi = w_multi / (1 - alpha_multi - beta_multi)
        st.write("The volatility forecast can be computed using the following formula:")
        st.latex(r'\sigma_{t+n, t}^2=\sigma^2+(\alpha+\beta)^{n-1}\left(\sigma_{t+1}^2-\sigma^2\right)'
                 r' \quad \text { for } \quad n \geq 1')
        nb_periods = st.number_input("Number of periods", min_value=0.0, max_value=100.0, value=5.0)
        sigma = st.number_input(r"$\sigma_{t+1}$", min_value=0.0, max_value=100.0, value=1.0)
        st.latex(rf'\sigma = {sigma_t_squared_multi:.8f}')
        st.latex(r"\sigma_{t+n, t}^2 = " + rf"{sigma_t_squared_multi + (alpha_multi + beta_multi)**(nb_periods-1) * (sigma**2 - sigma_t_squared_multi)}")
        st.latex(r"\sigma_{t+n, t} = " + rf"{(sigma_t_squared_multi + (alpha_multi + beta_multi)**(nb_periods-1) * (sigma**2 - sigma_t_squared_multi))**(1/2)}")
