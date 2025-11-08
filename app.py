import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from evaluate_performance import *

#Configuring the page
st.set_page_config(
    page_title="OptiTraders Portfolio Optimization Platform",
    layout="wide"
)

#Formatting! :)
st.markdown("""
    <style>
    /* Base styling */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'DejaVu Sans', sans-serif;
        text-align: center;
    }

    /* Center title and headers */
    h1, h2, h3, h4 {
        color: #F8F8F8;
        text-align: center;
    }

    /* Center markdown text and labels */
    .stMarkdown, div[data-testid="stMarkdownContainer"] p, label {
        color: #E7E9EC !important;
        text-align: center;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111319;
        color: #FFFFFF;
    }

    /* Buttons */
    div[data-testid="stButton"] > button {
        background-color: #1F2937;
        color: #FAFAFA;
        border-radius: 6px;
        padding: 0.45rem 0.85rem;
        border: none;
        transition: 0.3s ease;
        display: block;
        margin: 0 auto;
    }

    div[data-testid="stButton"] > button:hover {
        background-color: #374151;
        color: #FFFFFF;
    }

    /* Dropdowns (select/multiselect) */
    div[data-baseweb="select"] {
        background-color: transparent !important;
        color: #FAFAFA !important;
        border-radius: 6px;
        border: 1px solid #3E4450;
        display: block;
        margin: 0 auto;
    }

    /* Dropdown options */
    ul[data-baseweb="menu"] {
        background-color: #2E3440;
        color: #FAFAFA;
        border-radius: 6px;
        text-align: center;
    }

    /* Chart styling */
    .stPlotlyChart, .stPyplot {
        background-color: transparent !important;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>OptiTraders Portfolio Optimization Platform</h1>", unsafe_allow_html=True)


# Selecting the assets

st.markdown("<h3>Select assets for your portfolio:</h3>", unsafe_allow_html=True)
# tickers = ["AAPL", "GOOG", "AMZN", "MSFT", "TSLA", "META"]
close_data, dates, tickers = read_raw_csv('stock_details_5_years.csv')
selected_tickers = st.multiselect("label", options=tickers, default=tickers[:3],label_visibility='collapsed')
if not selected_tickers:
    st.warning("Please select at least one ticker to display the portfolio.")
else:
    inputs = get_inputs(close_data,dates,selected_tickers)

    # Making the split layout (2 columns) so the thing is divided and both are shown for easier comparison and
    # better looking visually

    col_mc, col_mk = st.columns(2, gap="large")

    #Monte carlo selection
    with col_mc:
        st.markdown("<h2>Monte Carlo Simulation</h2>", unsafe_allow_html=True)
        st.markdown(f"Sharpe Ratio: {inputs['Monte Carlo Sharpe']}")
        st.markdown(f"Weights: {np.array(inputs['Monte Carlo Weights']).tolist()}")
        st.markdown(f"Computation Time: {inputs['Monte Carlo Time']} seconds")
        st.markdown(f"Memory Usage: {inputs['Monte Carlo Memory']} kibibytes")

        st.markdown("<h3>Portfolio Visualization</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        colors_mc = ['red' if w < 0 else '#60A5FA' for w in inputs['Monte Carlo Weights']]
        ax.bar(selected_tickers, inputs['Monte Carlo Weights'], color=colors_mc)
        ax.set_ylim(np.min(inputs["Monte Carlo Weights"])-1, np.max(inputs["Monte Carlo Weights"])+1)
        ax.set_ylabel("Weight", color="#FAFAFA")
        ax.set_title("Monte Carlo Portfolio Weights", color="#FAFAFA", fontname='DejaVu Sans')
        ax.tick_params(colors="#FAFAFA")
        ax.axhline(0, color="#FAFAFA", linewidth=1.2)
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#1F2937')
        st.pyplot(fig)

    # Markowitz selection
    with col_mk:
        st.markdown("<h2>Markowitz Mean-Variance Optimization</h2>", unsafe_allow_html=True)
        st.markdown(f"Sharpe Ratio: {inputs['Markowitz Sharpe']}")
        st.markdown(f"Weights: {np.array(inputs['Markowitz Weights']).tolist()}")
        st.markdown(f"Computation Time: {inputs['Markowitz Time']} seconds")
        st.markdown(f"Memory Usage: {inputs['Markowitz Memory']} kibibytes")

        st.markdown("<h3>Portfolio Visualization</h3>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        colors_mk = ['red' if w < 0 else '#60A5FA' for w in inputs['Markowitz Weights']]
        ax2.bar(selected_tickers, inputs['Markowitz Weights'], color=colors_mk)
        ax2.set_ylim(np.min(inputs["Markowitz Weights"])-1, np.max(inputs["Markowitz Weights"])+1)
        ax2.set_ylabel("Weight", color="#FAFAFA")
        ax2.set_title("Markowitz Portfolio Weights", color="#FAFAFA", fontname='DejaVu Sans')
        ax2.tick_params(colors="#FAFAFA")
        ax2.axhline(0, color="#FAFAFA", linewidth=1.2)
        fig2.patch.set_facecolor('#0E1117')
        ax2.set_facecolor('#1F2937')
        st.pyplot(fig2)