import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

st.set_page_config(
    page_title="Discount Recommendation System",
    page_icon="🏷️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #ECEEF5 !important;
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden; }

.block-container {
    padding: 2.8rem 3rem 3rem !important;
    max-width: 1200px !important;
}

/* ── Sliders — stable, no jitter ── */
/* Remove ALL default Streamlit slider wrappers that cause layout shifts */
.stSlider > div { padding: 0 !important; }

.stSlider > label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
    display: block !important;
    margin-bottom: 0.15rem !important;
}

.stSlider {
    margin-bottom: 1.6rem !important;
    padding: 0 !important;
}

/* Track */
div[data-baseweb="slider"] > div:first-child {
    background: #D1D5E8 !important;
    height: 5px !important;
    border-radius: 99px !important;
}

/* Filled */
div[data-baseweb="slider"] div[role="progressbar"] {
    background: linear-gradient(90deg, #FB923C, #F97316) !important;
    border-radius: 99px !important;
}

/* Thumb — fixed size, no transform jitter */
div[data-baseweb="slider"] div[role="slider"] {
    background: #FFFFFF !important;
    border: 3px solid #F97316 !important;
    width: 20px !important;
    height: 20px !important;
    border-radius: 50% !important;
    box-shadow: 0 2px 10px rgba(249,115,22,0.3) !important;
    outline: none !important;
    /* No transform — prevents jitter */
}

/* Value tooltip */
div[data-baseweb="slider"] [data-testid="stTickBar"] { display: none !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #FB923C 0%, #F97316 60%, #EA580C 100%) !important;
    color: #FFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 0.82rem 2.5rem !important;
    border-radius: 14px !important;
    border: none !important;
    box-shadow: 0 4px 18px rgba(249,115,22,0.38) !important;
    transition: box-shadow 0.2s ease, transform 0.2s ease !important;
    margin-top: 0.4rem !important;
    width: auto !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(249,115,22,0.48) !important;
}

/* ── SHAP section titles ── */
.shap-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.15rem;
}
.shap-sub { font-size: 0.8rem; color: #9CA3AF; margin-bottom: 0.5rem; }

/* ── Result card ── */
.r-card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 1.7rem 1.8rem 1.5rem;
    box-shadow: 0 2px 16px rgba(15,23,42,0.07);
    border: 1px solid #E4E8F0;
    margin-bottom: 1rem;
}

.res-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    color: #111827;
    text-align: center;
    margin-bottom: 0.2rem;
}
.res-meta { font-size: 0.75rem; color: #9CA3AF; text-align: center; margin-bottom: 1.1rem; }

.conf-block {
    background: #F9FAFB;
    border: 1px solid #EAEDF3;
    border-radius: 12px;
    padding: 0.8rem 1rem 0.65rem;
    margin-bottom: 0.4rem;
}
.conf-row { display:flex; align-items:center; gap:0.65rem; margin-bottom:0.45rem; }
.conf-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.conf-lbl { font-size:0.84rem; font-weight:600; color:#1F2937; flex:1; }
.conf-track { background:#E5E7EB; height:6px; border-radius:99px; overflow:hidden; }
.conf-fill  { height:100%; border-radius:99px; background:linear-gradient(90deg,#22C55E,#F59E0B); }
.conf-pct-row { display:flex; justify-content:flex-end; margin-top:0.28rem; }
.conf-pct { font-size:0.8rem; font-weight:700; color:#374151; }
.conf-note { font-size:0.71rem; color:#9CA3AF; margin-top:0.35rem; }

/* ── Placeholder ── */
.placeholder {
    background: linear-gradient(135deg, #EFF6FF, #F0F9FF);
    border: 1.5px dashed #BAE6FD;
    border-radius: 14px;
    padding: 2rem 1.5rem;
    text-align: center;
}
.placeholder-text { font-size:0.86rem; font-weight:500; color:#0369A1; line-height:1.65; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<p style="font-family:'Plus Jakarta Sans',sans-serif;font-size:2rem;font-weight:800;
           color:#111827;letter-spacing:-0.5px;margin-bottom:0.25rem;">
    🏷️ Discount Recommendation System
</p>
<p style="font-size:0.9rem;color:#6B7280;margin-bottom:2.2rem;">
    Predict whether a customer should receive a discount based on their ordering behavior.
</p>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 1], gap="large")

# ══ LEFT ═════════════════════════════════════════════════════════════════════
with col_left:
    # Big "Customer Details" heading — pure HTML, never re-renders
    st.markdown("""
    <p style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.3rem;font-weight:700;
               color:#1F2937;margin-bottom:1.4rem;letter-spacing:-0.2px;">
        Customer Details
    </p>
    """, unsafe_allow_html=True)

    orders        = st.slider("Number of Orders Placed", 1, 100, 48)
    discount_pref = st.slider("Discount Preference  (1 = Low → 5 = High)", 1, 5, 4)
    order_value   = st.slider("Average Order Value (₹)", 100, 1000, 500, step=50)
    delivery_exp  = st.slider("Delivery Experience  (1 = Poor → 5 = Excellent)", 1, 5, 5)

    lm_d = {1:"Very Low", 2:"Low", 3:"Moderate", 4:"High", 5:"Very High"}
    lm_e = {1:"Poor", 2:"Fair", 3:"Good", 4:"Very Good", 5:"Excellent"}

    # Badges — keyed to slider values so they update correctly
    st.markdown(f"""
    <div style="display:flex;gap:0.45rem;flex-wrap:wrap;margin:0.4rem 0 1rem;">
        <span style="background:#FFF7ED;color:#C2410C;border:1px solid #FED7AA;
                     border-radius:8px;padding:0.27rem 0.7rem;font-size:0.78rem;font-weight:600;">
            📦 {orders} orders
        </span>
        <span style="background:#FFF7ED;color:#C2410C;border:1px solid #FED7AA;
                     border-radius:8px;padding:0.27rem 0.7rem;font-size:0.78rem;font-weight:600;">
            🏷️ {lm_d[discount_pref]}
        </span>
        <span style="background:#FFF7ED;color:#C2410C;border:1px solid #FED7AA;
                     border-radius:8px;padding:0.27rem 0.7rem;font-size:0.78rem;font-weight:600;">
            ₹{order_value} avg
        </span>
        <span style="background:#FFF7ED;color:#C2410C;border:1px solid #FED7AA;
                     border-radius:8px;padding:0.27rem 0.7rem;font-size:0.78rem;font-weight:600;">
            🚚 {lm_e[delivery_exp]}
        </span>
    </div>
    """, unsafe_allow_html=True)

    predict = st.button("Evaluate Customer for Discount")

# ══ RIGHT ════════════════════════════════════════════════════════════════════
with col_right:

    # ── Result ───────────────────────────────────────────────────────────────
    if predict:
        try:
            res = requests.post(
                "https://discount-recommendation-system.onrender.com/predict",
                json={"orders": orders, "discount": discount_pref,
                      "order_value": order_value, "delivery_exp": delivery_exp},
                timeout=5
            ).json()

            confidence = res.get("confidence", 0)
            recommend  = res.get("recommend_discount", 0)

            if recommend == 1:
                icon, dot_clr, title = "✅", "#22C55E", "Discount Recommended"
            else:
                icon, dot_clr, title = "❌", "#EF4444", "Discount Not Required"

            st.markdown(f"""
            <div class="r-card">
                <div style="text-align:center;margin-bottom:1rem;">
                    <div style="font-size:2.1rem;margin-bottom:0.4rem;">{icon}</div>
                    <div class="res-title">{title}</div>
                    <div class="res-meta">Result 75 · 70.0% · 13 minutes since yesterday</div>
                </div>
                <div class="conf-block">
                    <div class="conf-row">
                        <div class="conf-dot" style="background:{dot_clr};"></div>
                        <div class="conf-lbl">{title}</div>
                    </div>
                    <div class="conf-track">
                        <div class="conf-fill" style="width:{confidence*100:.0f}%;"></div>
                    </div>
                    <div class="conf-pct-row"><div class="conf-pct">{confidence*100:.0f}%</div></div>
                </div>
                <div class="conf-note">Local: DecisionTree &nbsp;|&nbsp; Local SHAP explanation not available.</div>
            </div>
            """, unsafe_allow_html=True)

        except requests.exceptions.RequestException:
            st.markdown("""
            <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:14px;
                        padding:1rem 1.1rem;color:#991B1B;font-size:0.85rem;font-weight:500;margin-bottom:1rem;">
                ⚠️ Cannot connect to prediction service. Make sure Flask API is running on port 5000.
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:14px;
                        padding:1rem 1.1rem;color:#991B1B;font-size:0.85rem;font-weight:500;margin-bottom:1rem;">
                ⚠️ Error: {str(e)}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="placeholder" style="margin-bottom:1rem;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">🔍</div>
            <div class="placeholder-text">
                Adjust the sliders and click<br>
                <strong>"Evaluate Customer for Discount"</strong><br>
                to see the recommendation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── SHAP chart — always visible, no wrapper card ──────────────────────────
    st.markdown('<div class="shap-title">Why did the model decide this?</div>', unsafe_allow_html=True)
    st.markdown('<div class="shap-sub">Top drivers influencing discount decisions across customers.</div>', unsafe_allow_html=True)

    path = "models/global_shap.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path).head(6)
            PALETTE = ["#F97316","#3B82F6","#8B5CF6","#EC4899","#14B8A6","#F59E0B"]

            fig, ax = plt.subplots(figsize=(4, 3.3))
            fig.patch.set_facecolor("none")
            ax.set_facecolor("none")

            _, _, autotexts = ax.pie(
                df["importance"],
                autopct="%1.1f%%",
                startangle=140,
                colors=PALETTE[:len(df)],
                pctdistance=0.72,
                wedgeprops={"linewidth": 2.5, "edgecolor": "#FFF"},
            )
            for at in autotexts:
                at.set_color("white"); at.set_fontsize(7.2); at.set_weight("bold")

            ax.add_patch(plt.Circle((0, 0), 0.42, fc="white"))

            patches = [mpatches.Patch(color=PALETTE[i], label=df["feature"].iloc[i]) for i in range(len(df))]
            ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.18),
                      ncol=3, fontsize=6.5, frameon=False, handlelength=0.9,
                      handleheight=0.8, labelcolor="#374151")

            plt.tight_layout()
            st.pyplot(fig, width='stretch')
            plt.close()

        except Exception as e:
            st.markdown(f'<div class="placeholder"><div class="placeholder-text">⚠️ {str(e)}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="placeholder">
            <div style="font-size:1.5rem;margin-bottom:0.35rem;">📊</div>
            <div class="placeholder-text">
                SHAP data not found at<br>
                <code style="font-size:0.76rem;">models/global_shap.csv</code>
            </div>
        </div>
        """, unsafe_allow_html=True)