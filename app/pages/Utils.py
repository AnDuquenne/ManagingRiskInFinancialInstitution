import streamlit as st
import numpy as np
import sympy
import pyperclip

tab_volatility_converter, return_computation, math_comp = st.tabs(["Volatility Converter", "Return computation",
                                                                   "Mathematical computation"])

with tab_volatility_converter:
    with st.expander("Year to day volatility converter"):
        in_1 = st.number_input("Yearly volatility (in decimal value)", min_value=0.0, max_value=100.0, value=0.10)
        st.write(rf"Daily volatility: {in_1 / np.sqrt(252):.8f}")
        # Button to copy the value
        if st.button("Copy to Clipboard", key="year_to_day"):
            pyperclip.copy(in_1 / np.sqrt(252))
            st.success("Value copied to clipboard!")

    with st.expander("Day to year volatility converter"):
        in_2 = st.number_input("Daily volatility (in decimal value)", min_value=0.0, max_value=100.0, value=0.10)
        st.write(rf"Yearly volatility: {in_2 * np.sqrt(252):.8f}")
        # Button to copy the value
        if st.button("Copy to Clipboard", key="day_to_year"):
            pyperclip.copy(in_2 * np.sqrt(252))
            st.success("Value copied to clipboard!")

with return_computation:
    st.write("Return computation")
    in_3 = st.number_input("Value at t1", min_value=0.0, max_value=1000.0, value=100.0)
    in_4 = st.number_input("Value at t2", min_value=0.0, max_value=1000.0, value=102.0)
    return_ = (in_4-in_3)/in_3
    st.write(rf"Return: {(in_4-in_3)/in_3:.8f}")
    # Button to copy the value
    if st.button("Copy to Clipboard", key="return"):
        pyperclip.copy(return_)
        st.success("Value copied to clipboard!")

with math_comp:
    expression = st.text_input("Enter a math expression (e.g. sqrt(2), (1+2)*3, sin(pi/2)):")

    if expression:
        try:
            # Parse the expression into a Sympy expression
            sym_expr = sympy.sympify(expression)
            # Evaluate it numerically
            result = sym_expr.evalf()
            st.write(f"**Expression:** {sym_expr}")
            st.write(f"**Result:** {result}")
        except Exception as e:
            st.error(f"Error evaluating expression: {e}")

        # Button to copy the value
        if st.button("Copy to Clipboard", key="math_comp"):
            pyperclip.copy(result)
            st.success("Value copied to clipboard!")