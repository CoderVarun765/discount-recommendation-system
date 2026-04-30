import streamlit as st 
import streamlit.components.v1 as components
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
import requests



st.set_page_config(
    page_title="Discount Recommendation System",
    layout="wide"
)


def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.markdown(
    """
    <div class="title-wrapper">
        <h1 class="project-title">Discount Recommendation System</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Parent + Child Card (Iframe DOM)
components.html(
"""
<style>
.page-wrapper {
    display: flex;
    justify-content: center;
}

.parent-card {
    background: #e5e7eb;
    width: 100%;
    max-width: 900px;
    min-height: 320px;
    padding: 32px;
    border-radius: 16px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.08);
}

/* Main inner layout */
.parent-inner {
    display: flex;
    gap: 20px;
}

/* LEFT CARD */
.child-left {
    width: 50%;
    background: white;
    border-radius: 16px;
    padding: 24px;
}

/* RIGHT CONTAINER */
.child-right {
    width: 50%;
    display: flex;
    flex-direction: column;
    gap: 20px;
}

/* RIGHT CARDS */
.child-right-card {
    flex: 1;
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
}
</style>

<div class="page-wrapper">
  <div class="parent-card">

    <div class="parent-inner">

      <!-- LEFT CARD -->
      <div class="child-left">
        <h3>Main Dashboard</h3>
        <p>
          Predict whether a customer should receive a discount
          based on their ordering behaviour.
        </p>
           <div class= "customer-details">
              <h2>Customer Details</h2>  
           </div>
      </div>

      <!-- RIGHT SIDE -->
      <div class="child-right">

        <div class="child-right-card">
          <h4>Model Status</h4>
          <p>
            View model accuracy, version, and health.
          </p>
        </div>

        <div class="child-right-card">
          <h4>User Insights</h4>
          <p>
            Analyze customer behavior and segments.
          </p>
        </div>

      </div>

    </div>

  </div>
</div>
""",
height=380
)

