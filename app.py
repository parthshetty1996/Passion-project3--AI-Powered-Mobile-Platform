"""
FinWise AI — Streamlit Analytics Dashboard
==========================================
SP Jain GMBA | Data Analytics | Dubai 2026
"""

import os, json, pickle, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, roc_curve,
                              f1_score, precision_score, recall_score,
                              r2_score, mean_absolute_error, mean_squared_error)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinWise AI — Analytics Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Colours ──────────────────────────────────────────────────────────────────
NAVY  = "#1B3A6B"
TEAL  = "#0A7C8C"
GOLD  = "#C8A84B"
CORAL = "#E07B5A"
SAGE  = "#5A8A6B"
PALETTE = [NAVY, TEAL, GOLD, CORAL, SAGE, "#8B5CF6"]
PERSONA_COLORS = {
    "Tech-Savvy Millennial": TEAL,
    "Struggling Expat":      CORAL,
    "Senior Finance Pro":    NAVY,
    "High-Earning Nomad":    GOLD,
    "Cautious Saver":        SAGE,
}

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main {background-color:#F8FAFC;}
.block-container {padding-top:1.5rem;padding-bottom:2rem;}
h1 {color:#1B3A6B;font-weight:800;}
h2 {color:#1B3A6B;font-weight:700;border-bottom:2px solid #0A7C8C;padding-bottom:6px;}
h3 {color:#0A7C8C;font-weight:600;}
.insight-box{background:#EBF5F7;border-radius:8px;padding:14px 16px;
             border-left:4px solid #C8A84B;margin:10px 0;font-size:.93rem;}
.metric-card{background:white;border-radius:10px;padding:18px 20px;
             border-left:4px solid #0A7C8C;box-shadow:0 2px 6px rgba(0,0,0,.07);
             margin-bottom:12px;}
div[data-testid="metric-container"]{background:white;border-radius:8px;
             padding:14px;box-shadow:0 2px 5px rgba(0,0,0,.06);}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PATH RESOLUTION  — works on local, Streamlit Cloud, Colab
# ══════════════════════════════════════════════════════════════════════════════
HERE = Path(__file__).resolve().parent          # folder that contains app.py

FILENAMES = {
    "master":  "FinWise_Master_Cleaned.csv",
    "clf":     "FinWise_Classification_Dataset.csv",
    "clust":   "FinWise_Clustering_Dataset.csv",
    "assoc":   "FinWise_Association_Dataset.csv",
    "reg":     "FinWise_Regression_Dataset.csv",
}

def find_csv(filename: str) -> Path:
    """Try data/ subfolder first, then repo root, then cwd."""
    for candidate in [
        HERE / "data" / filename,
        HERE / filename,
        Path.cwd() / "data" / filename,
        Path.cwd() / filename,
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find {filename}.\n"
        f"Looked in:\n"
        f"  {HERE / 'data' / filename}\n"
        f"  {HERE / filename}\n"
        f"Please upload it using the sidebar uploader."
    )

@st.cache_data
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load(key: str) -> pd.DataFrame:
    """Load from session_state (uploaded) or disk."""
    if key in st.session_state:
        return st.session_state[key]
    path = find_csv(FILENAMES[key])
    return read_csv(str(path))

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:16px 0;'>
      <div style='font-size:2.2rem;'>💳</div>
      <div style='color:{NAVY};font-size:1.3rem;font-weight:800;'>FinWise AI</div>
      <div style='color:{TEAL};font-size:.85rem;'>Analytics Dashboard</div>
      <hr style='border-color:{TEAL};margin:10px 0;'>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📂 Upload CSVs")
    st.caption("Only needed if files are missing on the server.")

    UPLOAD_LABELS = {
        "master": "FinWise_Master_Cleaned.csv",
        "clf":    "FinWise_Classification_Dataset.csv",
        "clust":  "FinWise_Clustering_Dataset.csv",
        "assoc":  "FinWise_Association_Dataset.csv",
        "reg":    "FinWise_Regression_Dataset.csv",
    }
    with st.expander("📤 Upload datasets (optional)"):
        for key, label in UPLOAD_LABELS.items():
            uf = st.file_uploader(label, type=["csv"], key=f"uf_{key}")
            if uf:
                st.session_state[key] = pd.read_csv(uf)
                st.success(f"✅ {key} loaded")

    st.markdown("---")
    st.markdown(f"""<div style='font-size:.82rem;color:#666;'>
    <b>SP Jain GMBA</b><br>Data Analytics | Term 2<br>Dubai Campus, 2026
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ALL DATASETS
# ══════════════════════════════════════════════════════════════════════════════
try:
    master   = load("master")
    clf_df   = load("clf")
    clust_df = load("clust")
    assoc_df = load("assoc")
    reg_df   = load("reg")
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Upload the 5 CSV files using the sidebar **Upload datasets** expander, then refresh.")
    st.stop()

with st.sidebar:
    st.success(f"✅ {len(master):,} respondents loaded")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER + KPIs
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='background:linear-gradient(135deg,{NAVY},{TEAL});border-radius:12px;
            padding:24px 30px;margin-bottom:24px;'>
  <h1 style='color:white;margin:0;font-size:1.8rem;'>
    💳 FinWise AI — Business Feasibility & Market Analytics
  </h1>
  <p style='color:#B8DDE0;margin:6px 0 0;font-size:.95rem;'>
    AI-Powered Expense Tracking · Multi-Currency Payments · Financial Wellness · Dubai, UAE
  </p>
</div>""", unsafe_allow_html=True)

wtp_pct  = master["Q20_WTP_Binary"].mean() * 100
avg_wtp  = master["Q21_WTP_AED"].mean()
avg_dl   = master["Q24_DownloadLikelihood"].mean()
avg_nps  = master["Q29_NPS"].mean()
pain_avg = master["Derived_PainComposite"].mean()

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("📊 Willing to Pay",  f"{wtp_pct:.1f}%",   f"+{wtp_pct-50:.1f}% above neutral")
k2.metric("💰 Avg WTP",        f"AED {avg_wtp:.0f}", f"Median AED {master['Q21_WTP_AED'].median():.0f}")
k3.metric("📱 Download Score", f"{avg_dl:.1f}/10",   f"σ={master['Q24_DownloadLikelihood'].std():.1f}")
k4.metric("🔥 Pain Score",     f"{pain_avg:.2f}/5",  "Moderate-High demand")
k5.metric("⭐ NPS",            f"{avg_nps:.1f}/10",  f"{master['Q29_NPS_Category'].value_counts().get('Promoter',0)} Promoters")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 EDA & Descriptive",
    "🔍 Diagnostic & Cross-tab",
    "🤖 Classification",
    "🔵 Clustering",
    "🔗 Association Rules",
    "📈 Regression",
    "🏆 Business Findings"
])

# ─── TAB 1: EDA ───────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Exploratory Data Analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Willingness to Pay")
        wtp_c = master["Q20_WTP_Binary"].value_counts().reset_index()
        wtp_c.columns = ["WTP","Count"]
        wtp_c["Label"] = wtp_c["WTP"].map({1:"Willing",0:"Not Willing"})
        fig = px.pie(wtp_c, names="Label", values="Count", hole=0.55,
                     color="Label", color_discrete_map={"Willing":TEAL,"Not Willing":CORAL})
        fig.update_layout(height=300, margin=dict(t=20,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""<div class='insight-box'>
        <b>{wtp_pct:.1f}%</b> willing to pay — exceeds the 50% SaaS benchmark, confirming
        strong product-market fit for FinWise AI.</div>""", unsafe_allow_html=True)

    with c2:
        st.subheader("Download Likelihood (1–10)")
        dl_c = master["Q24_DownloadLikelihood"].value_counts().sort_index().reset_index()
        dl_c.columns = ["Score","Count"]
        dl_c["Group"] = dl_c["Score"].apply(
            lambda x: "High (7-10)" if x>=7 else "Moderate (5-6)" if x>=5 else "Low (<5)")
        fig = px.bar(dl_c, x="Score", y="Count", color="Group",
                     color_discrete_map={"High (7-10)":TEAL,"Moderate (5-6)":GOLD,"Low (<5)":CORAL})
        fig.add_vline(x=avg_dl, line_dash="dash", line_color=NAVY,
                      annotation_text=f"Mean={avg_dl:.1f}")
        fig.update_layout(height=300, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Demographics")
    demo_choice = st.selectbox("Dimension:", [
        "Age Group","Education","Income Tier","Employment","Years in Dubai"])
    demo_map = {
        "Age Group":    ("Q1_AgeGroup",  {1:"<25",2:"25-34",3:"35-44",4:"45-54",5:"55+"}),
        "Education":    ("Q3_Education", {1:"High School",2:"Bachelor's",3:"Master/MBA",4:"PhD",5:"Prof.Qual"}),
        "Income Tier":  ("Q5_IncomeTier",{1:"<5k",2:"5-10k",3:"10-20k",4:"20-40k",5:">40k",6:"PNS"}),
        "Employment":   ("Q4_Employment",{1:"Private",2:"Govt",3:"Freelance",4:"Business",5:"Part-time"}),
        "Years in Dubai":("Q7_YearsDubai",{1:"<1yr",2:"1-3yr",3:"3-5yr",4:"5-10yr",5:">10yr"}),
    }
    col, mapping = demo_map[demo_choice]
    dc = master[col].map(mapping).value_counts().reset_index()
    dc.columns = ["Category","Count"]
    dc["Pct"] = (dc["Count"]/len(master)*100).round(1)
    fig = px.bar(dc, x="Category", y="Count", text="Pct",
                 color="Count", color_continuous_scale=["#B8DDE0",NAVY])
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=340, showlegend=False, coloraxis_showscale=False,
                      margin=dict(t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pain Point Scores (Q13 — Mean 1-5)")
    pain_map = {
        "Q13_DiffTrackMultiCurrency":"Multi-Currency Tracking",
        "Q13_LoseFXMoney":"FX Money Loss",
        "Q13_SavingsHabit":"Savings Habit",
        "Q13_MultiBankTimeConsuming":"Multi-Bank Mgmt",
        "Q13_DissatisifiedTools":"Tool Dissatisfaction",
        "Q13_UnexpectedExpenses":"Unexpected Expenses",
        "Q13_HealthUnderstanding":"Financial Health Clarity",
        "Q13_Overspent":"Overspending"
    }
    pm = master[list(pain_map.keys())].mean().rename(pain_map).sort_values()
    fig = go.Figure(go.Bar(
        x=pm.values, y=pm.index, orientation="h",
        marker_color=[CORAL if v>=3.5 else GOLD if v>=3.0 else SAGE for v in pm.values],
        text=[f"{v:.2f}" for v in pm.values], textposition="outside"
    ))
    fig.add_vline(x=3.0, line_dash="dot", line_color="gray", annotation_text="Neutral (3.0)")
    fig.update_layout(height=360, margin=dict(t=20,b=10,l=200,r=80),
                      xaxis=dict(range=[0,5.5]))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Descriptive Statistics Table"):
        desc_cols = ["Q1_AgeRaw","Q21_WTP_AED","Q24_DownloadLikelihood","Q29_NPS",
                     "Derived_PainComposite","Derived_TechAffinity","Derived_AdoptionAttitude",
                     "Derived_FinancialStress","Eng_PainGap","Eng_DigitalEngagement"]
        st.dataframe(master[desc_cols].describe().round(3), use_container_width=True)

# ─── TAB 2: DIAGNOSTIC ────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Diagnostic Analysis & Cross-Tabulation")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("WTP % by Persona")
        wtp_p = pd.crosstab(master["Persona_Cluster"],
                            master["Q20_WTP_Binary"].map({0:"Not Willing",1:"Willing"}),
                            normalize="index") * 100
        if "Willing" in wtp_p.columns:
            fig = px.bar(wtp_p.reset_index(), x="Persona_Cluster", y="Willing",
                         color="Persona_Cluster", color_discrete_map=PERSONA_COLORS,
                         text=wtp_p["Willing"].round(1).astype(str)+"  %")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=360, showlegend=False, yaxis_range=[0,105],
                               xaxis_title="", yaxis_title="% Willing to Pay",
                               margin=dict(t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("WTP % by Income Tier")
        inc_map = {1:"<5k",2:"5-10k",3:"10-20k",4:"20-40k",5:">40k",6:"PNS"}
        wtp_i = pd.crosstab(master["Q5_IncomeTier"].map(inc_map),
                            master["Q20_WTP_Binary"].map({0:"Not Willing",1:"Willing"}),
                            normalize="index") * 100
        order = [v for v in ["<5k","5-10k","10-20k","20-40k",">40k","PNS"] if v in wtp_i.index]
        wtp_i = wtp_i.reindex(order)
        if "Willing" in wtp_i.columns:
            fig = px.bar(wtp_i.reset_index(), x="Q5_IncomeTier", y="Willing",
                         text=wtp_i["Willing"].round(1).astype(str)+"  %",
                         color="Willing", color_continuous_scale=[CORAL,TEAL])
            fig.update_traces(textposition="outside")
            fig.update_layout(height=360, coloraxis_showscale=False,
                               xaxis_title="Income (AED/mo)", yaxis_title="% Willing",
                               yaxis_range=[0,105], margin=dict(t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix")
    corr_opts = st.multiselect("Variables:", [
        "Derived_PainComposite","Derived_TechAffinity","Derived_AdoptionAttitude",
        "Derived_FinancialStress","Q5_IncomeTier","Q10_NumCurrencies",
        "Q24_DownloadLikelihood","Q29_NPS","Q21_WTP_AED","Q20_WTP_Binary",
        "Eng_PainGap","Eng_DigitalEngagement","Q1_AgeRaw","Q27_CurrentSatisfaction"
    ], default=["Derived_PainComposite","Q21_WTP_AED","Q24_DownloadLikelihood",
                "Derived_TechAffinity","Q20_WTP_Binary","Eng_PainGap"])
    if len(corr_opts) >= 3:
        fig = px.imshow(master[corr_opts].corr(), text_auto=".2f",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=400, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pain Composite vs WTP by Persona")
    r, _ = stats.pearsonr(master["Derived_PainComposite"], master["Q21_WTP_AED"])
    fig = px.scatter(master, x="Derived_PainComposite", y="Q21_WTP_AED",
                     color="Persona_Cluster", color_discrete_map=PERSONA_COLORS,
                     opacity=0.5,
                     labels={"Derived_PainComposite":"Pain Score","Q21_WTP_AED":"WTP (AED/mo)"})
    # Manual OLS trendline using numpy (no statsmodels dependency)
    x_vals = master["Derived_PainComposite"].values
    y_vals = master["Q21_WTP_AED"].values
    m, b = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    fig.add_trace(go.Scatter(x=x_line, y=m * x_line + b,
                              mode="lines", name=f"Trend (r={r:.2f})",
                              line=dict(color="black", width=2, dash="dash")))
    fig.update_layout(height=400, title=f"Pearson r = {r:.3f}")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"""<div class='insight-box'>
    Pain and WTP are positively correlated (r = {r:.3f}) — respondents who feel more
    financial friction are willing to pay more. This validates FinWise AI's core value proposition.
    </div>""", unsafe_allow_html=True)

# ─── TAB 3: CLASSIFICATION ────────────────────────────────────────────────────
with tabs[2]:
    st.header("Classification — Predicting Willingness to Pay")

    LEAKAGE_CLF = {"Q20_WTP_Binary","Q20_WTP_Intent","Q21_WTP_AED","Q21_WTP_Tier",
                   "Eng_WTPperPain","Eng_AdoptionReadiness","Respondent_ID",
                   "Persona_Cluster","Persona_Cluster_Enc","Q29_NPS_Category"}

    @st.cache_resource
    def train_classifiers(_df):
        feat_cols = [c for c in _df.columns
                     if c not in LEAKAGE_CLF and not c.endswith("_Label")
                     and pd.api.types.is_numeric_dtype(_df[c])]
        X = _df[feat_cols].fillna(0)
        y = _df["Q20_WTP_Binary"]
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        sc = StandardScaler()
        Xtr_sc = sc.fit_transform(Xtr)
        Xte_sc  = sc.transform(Xte)
        models = {
            "KNN":           KNeighborsClassifier(n_neighbors=9),
            "Decision Tree": DecisionTreeClassifier(max_depth=7,min_samples_split=20,
                                                    min_samples_leaf=10,random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200,max_depth=10,
                                                    min_samples_leaf=5,random_state=42,n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=150,max_depth=5,
                                                            learning_rate=0.08,random_state=42),
        }
        out = {}
        for name, m in models.items():
            Xtr_use = Xtr_sc if name=="KNN" else Xtr
            Xte_use = Xte_sc if name=="KNN" else Xte
            m.fit(Xtr_use, ytr)
            yp = m.predict(Xte_use)
            ypr = m.predict_proba(Xte_use)[:,1] if hasattr(m,"predict_proba") else None
            out[name] = {
                "train_acc": accuracy_score(ytr, m.predict(Xtr_use)),
                "test_acc":  accuracy_score(yte, yp),
                "precision": precision_score(yte, yp),
                "recall":    recall_score(yte, yp),
                "f1":        f1_score(yte, yp),
                "auc":       roc_auc_score(yte,ypr) if ypr is not None else None,
                "cm":        confusion_matrix(yte,yp).tolist(),
                "yte":       yte.tolist(),
                "ypr":       ypr.tolist() if ypr is not None else None,
            }
        rf = models["Random Forest"]
        fi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False).head(15)
        return out, fi

    with st.spinner("Training classifiers…"):
        clf_res, fi_series = train_classifiers(clf_df)

    # Performance table
    st.subheader("Model Performance")
    perf = pd.DataFrame({n:{
        "Train Acc":f"{v['train_acc']:.3f}","Test Acc":f"{v['test_acc']:.3f}",
        "Precision":f"{v['precision']:.3f}","Recall":f"{v['recall']:.3f}",
        "F1":f"{v['f1']:.3f}","AUC":(f"{v['auc']:.3f}" if v['auc'] is not None else "N/A")
    } for n,v in clf_res.items()}).T.reset_index().rename(columns={"index":"Model"})
    st.dataframe(perf.reset_index(drop=True), use_container_width=True)

    best_m = max(clf_res.items(), key=lambda x: x[1]["f1"])[0]
    _best_auc = clf_res[best_m]['auc']
    _auc_str  = f"{_best_auc:.3f}" if _best_auc is not None else "N/A"
    st.markdown(f"""<div class='insight-box'>
    <b>🏆 Best Model: {best_m}</b> — F1={clf_res[best_m]['f1']:.3f}, AUC={_auc_str}
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        sel_m = st.selectbox("Confusion matrix for:", list(clf_res.keys()), index=2)
        cm_arr = np.array(clf_res[sel_m]["cm"])
        fig, ax = plt.subplots(figsize=(5,4), facecolor="white")
        sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Not Willing","Willing"],
                    yticklabels=["Not Willing","Willing"], annot_kws={"size":13})
        ax.set_title(f"Confusion Matrix — {sel_m}", fontsize=11, color=NAVY, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        st.subheader("ROC Curves")
        fig = go.Figure()
        roc_colors = [NAVY,TEAL,GOLD,CORAL]
        for (n,v),col in zip(clf_res.items(), roc_colors):
            if v["ypr"]:
                fpr,tpr,_ = roc_curve(v["yte"],v["ypr"])
                fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",
                                         name=f"{n} (AUC={v['auc']:.2f})" if v['auc'] is not None else f"{n}",
                                         line=dict(color=col,width=2)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",
                                  line=dict(color="gray",dash="dash")))
        fig.update_layout(height=380,xaxis_title="FPR",yaxis_title="TPR",
                           margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance — Random Forest")
    fi_df = fi_series.reset_index()
    fi_df.columns = ["Feature","Importance"]
    fi_df["Type"] = fi_df["Feature"].apply(
        lambda x: "Engineered" if x.startswith("Eng_") else
                  "Derived" if x.startswith("Derived_") else "Survey Item")
    fig = px.bar(fi_df.sort_values("Importance"), x="Importance", y="Feature",
                 orientation="h", color="Type",
                 color_discrete_map={"Engineered":TEAL,"Derived":GOLD,"Survey Item":NAVY})
    fig.update_layout(height=460, margin=dict(t=20,b=10,l=240))
    st.plotly_chart(fig, use_container_width=True)

# ─── TAB 4: CLUSTERING ────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Clustering — Customer Segment Discovery")

    @st.cache_resource
    def run_clustering(_df):
        num_cols = [c for c in _df.select_dtypes(include=[np.number]).columns
                    if "Persona_Cluster_Enc" not in c]
        X = _df[num_cols].fillna(0)
        sc = StandardScaler()
        Xsc = sc.fit_transform(X)
        km = KMeans(n_clusters=5, random_state=42, n_init=20)
        labels = km.fit_predict(Xsc)
        sil = silhouette_score(Xsc, labels, sample_size=600, random_state=42)
        pca = PCA(n_components=2, random_state=42)
        Xpca = pca.fit_transform(Xsc)
        return labels, sil, Xpca, pca.explained_variance_ratio_

    with st.spinner("Running K-Means…"):
        labels, sil, Xpca, var_exp = run_clustering(clust_df)

    cdf = clust_df.copy()
    cdf["Cluster"] = labels

    prof = cdf.groupby("Cluster").agg({
        "Derived_PainComposite":"mean","Derived_TechAffinity":"mean",
        "Q21_WTP_AED":"mean","Q1_AgeRaw":"mean",
        "Q5_IncomeTier_Imputed":"mean","Eng_AdoptionReadiness":"mean"
    }).round(2)

    pain_r   = prof["Derived_PainComposite"].rank().astype(int)
    tech_r   = prof["Derived_TechAffinity"].rank().astype(int)
    inc_r    = prof["Q5_IncomeTier_Imputed"].rank().astype(int)
    age_r    = prof["Q1_AgeRaw"].rank().astype(int)
    name_map = {}
    for cid in range(5):
        if pain_r[cid]==5 and inc_r[cid]<=2:
            name_map[cid]="Budget-Conscious Expats"
        elif inc_r[cid]>=4 and tech_r[cid]>=3:
            name_map[cid]="High-Value Nomads"
        elif tech_r[cid]==5 and age_r[cid]<=2:
            name_map[cid]="Digital-First Millennials"
        elif pain_r[cid]<=2 and inc_r[cid]>=3:
            name_map[cid]="Finance Professionals"
        else:
            name_map[cid]="Cautious Savers"

    cdf["Cluster_Name"] = cdf["Cluster"].map(name_map)
    CCOLORS = {"Budget-Conscious Expats":CORAL,"High-Value Nomads":GOLD,
               "Digital-First Millennials":TEAL,"Finance Professionals":NAVY,
               "Cautious Savers":SAGE}

    c1,c2 = st.columns([3,2])
    with c1:
        st.subheader(f"PCA 2D Map (Silhouette={sil:.3f})")
        pdf = pd.DataFrame({"PC1":Xpca[:,0],"PC2":Xpca[:,1],"Cluster":cdf["Cluster_Name"]})
        fig = px.scatter(pdf, x="PC1", y="PC2", color="Cluster",
                         color_discrete_map=CCOLORS, opacity=0.55,
                         labels={"PC1":f"PC1 ({var_exp[0]*100:.0f}% var)",
                                  "PC2":f"PC2 ({var_exp[1]*100:.0f}% var)"})
        fig.update_traces(marker_size=5)
        fig.update_layout(height=420, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Cluster Profiles")
        pn = cdf.groupby("Cluster_Name").agg(
            N=("Cluster","count"), Pain=("Derived_PainComposite","mean"),
            Tech=("Derived_TechAffinity","mean"), WTP=("Q21_WTP_AED","mean"),
            Readiness=("Eng_AdoptionReadiness","mean"), Age=("Q1_AgeRaw","mean")
        ).round(2)
        st.dataframe(pn.style.background_gradient(subset=["WTP"],cmap="Greens")
                              .background_gradient(subset=["Pain"],cmap="Reds"),
                     use_container_width=True)

    st.subheader("Mean WTP by Cluster")
    wc = cdf.groupby("Cluster_Name")["Q21_WTP_AED"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(wc, x="Cluster_Name", y="Q21_WTP_AED", color="Cluster_Name",
                 color_discrete_map=CCOLORS,
                 text=wc["Q21_WTP_AED"].round(0).astype(int).astype(str).radd("AED "))
    fig.update_traces(textposition="outside")
    fig.update_layout(height=360, showlegend=False, yaxis_range=[0,105],
                       xaxis_title="", yaxis_title="Mean WTP (AED/month)",
                       margin=dict(t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Descriptions")
    descs = {
        "High-Value Nomads":        ("🌍","Senior expats, high income, multi-currency needs. Low pain but highest WTP (~AED 80/mo). Best premium tier target."),
        "Digital-First Millennials":("📱","Young tech-savvy pros (avg age 31). Highest tech affinity. Strong freemium-to-paid conversion potential."),
        "Finance Professionals":    ("📊","Experienced, analytically sophisticated. Low pain, moderate WTP. Need advanced investment & reporting features."),
        "Budget-Conscious Expats":  ("💸","Highest pain, lowest income. Core free-tier users. Convert after demonstrating savings ROI."),
        "Cautious Savers":          ("🏦","Privacy-conscious, moderate income. Need regulatory endorsement & data-security proof points to convert."),
    }
    cols = st.columns(5)
    for col,(name,(icon,desc)) in zip(cols,descs.items()):
        wtp_val = pn.loc[name,"WTP"] if name in pn.index else 0
        with col:
            st.markdown(f"""
            <div style='background:white;border-radius:10px;padding:14px 10px;
                        border-top:4px solid {CCOLORS.get(name,NAVY)};
                        box-shadow:0 2px 6px rgba(0,0,0,.07);'>
              <div style='text-align:center;font-size:1.4rem;'>{icon}</div>
              <div style='font-weight:700;color:{NAVY};font-size:.82rem;
                          text-align:center;margin:4px 0;'>{name}</div>
              <div style='text-align:center;color:{TEAL};font-weight:700;'>
                AED {wtp_val:.0f}/mo</div>
              <hr style='margin:6px 0;'>
              <div style='font-size:.76rem;color:#555;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

# ─── TAB 5: ASSOCIATION RULES ─────────────────────────────────────────────────
with tabs[4]:
    st.header("Association Rule Mining")

    @st.cache_resource
    def run_apriori(_df, sup, conf, lift):
        freq = apriori(_df.astype(bool), min_support=sup, use_colnames=True, max_len=4)
        rules = association_rules(freq, metric="lift", min_threshold=lift)
        rules = rules[rules["confidence"] >= conf]
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
        return rules.sort_values("lift", ascending=False)

    c1,c2,c3 = st.columns(3)
    sup  = c1.slider("Min Support",  0.05,0.30,0.10,0.01)
    conf = c2.slider("Min Confidence",0.40,0.90,0.55,0.05)
    lift = c3.slider("Min Lift",      1.0, 2.0, 1.1, 0.05)

    with st.spinner("Mining rules…"):
        rules_df = run_apriori(assoc_df, sup, conf, lift)

    st.markdown(f"**{len(rules_df):,} rules** found (support≥{sup}, confidence≥{conf}, lift≥{lift})")

    c1,c2 = st.columns([3,2])
    with c1:
        fig = px.scatter(rules_df.head(500), x="support", y="confidence",
                         color="lift", size="lift",
                         color_continuous_scale="RdYlGn",
                         hover_data=["antecedents_str","consequents_str","lift"])
        fig.update_layout(height=400, margin=dict(t=20,b=10),
                           title="Support vs Confidence (colour = Lift)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        top = rules_df.head(15)[["antecedents_str","consequents_str",
                                   "support","confidence","lift"]].copy()
        top.columns = ["Antecedents","Consequents","Supp","Conf","Lift"]
        top["Supp"]=top["Supp"].round(3); top["Conf"]=top["Conf"].round(3)
        top["Lift"]=top["Lift"].round(3)
        st.dataframe(top, use_container_width=True, height=400)

    wtp_rules = rules_df[rules_df["consequents_str"].str.contains("WTP|ValueSeeker|HighPain", na=False)]
    if len(wtp_rules):
        st.subheader("Rules predicting WTP / High Pain")
        wr = wtp_rules.head(10)[["antecedents_str","consequents_str","confidence","lift"]].copy()
        wr.columns = ["Conditions","Outcome","Confidence","Lift"]
        wr["Confidence"]=wr["Confidence"].round(3); wr["Lift"]=wr["Lift"].round(3)
        st.dataframe(wr, use_container_width=True)
        st.markdown(f"""<div class='insight-box'>
        <b>{len(wtp_rules)} rules</b> predict WTP or high-pain status. Remittance usage
        + FX concerns + subscription tracking consistently co-occur with payment willingness.
        </div>""", unsafe_allow_html=True)

# ─── TAB 6: REGRESSION ────────────────────────────────────────────────────────
with tabs[5]:
    st.header("Regression — Predicting WTP Amount")

    LEAKAGE_REG = {"Q21_WTP_AED","Q24_DownloadLikelihood","Respondent_ID",
                   "Eng_WTPperPain","Eng_AdoptionReadiness",
                   "Q20_WTP_Binary","Q20_WTP_Intent","Q21_WTP_Tier",
                   "Persona_Cluster","Persona_Cluster_Enc","Q29_NPS_Category"}

    @st.cache_resource
    def train_regressors(_df):
        fcols = [c for c in _df.columns
                 if c not in LEAKAGE_REG and not c.endswith("_Label")
                 and pd.api.types.is_numeric_dtype(_df[c])]
        X = _df[fcols].fillna(0)
        sc = StandardScaler()
        Xsc = sc.fit_transform(X)
        out = {}
        for tgt in ["Q21_WTP_AED","Q24_DownloadLikelihood"]:
            if tgt not in _df.columns:
                continue
            y = _df[tgt]
            Xtr,Xte,ytr,yte = train_test_split(Xsc,y,test_size=0.2,random_state=42)
            ms = {"Linear":LinearRegression(),"Ridge":Ridge(alpha=1.0),
                  "RF":RandomForestRegressor(n_estimators=150,max_depth=8,random_state=42,n_jobs=-1),
                  "GB":GradientBoostingRegressor(n_estimators=150,max_depth=5,learning_rate=0.08,random_state=42)}
            out[tgt]={}
            for mn,m in ms.items():
                m.fit(Xtr,ytr); yp=m.predict(Xte)
                out[tgt][mn]={"r2":r2_score(yte,yp),"mae":mean_absolute_error(yte,yp),
                               "rmse":np.sqrt(mean_squared_error(yte,yp)),
                               "yte":yte.tolist(),"ypred":yp.tolist()}
            rf=ms["RF"]
            out[tgt]["fi"]=pd.Series(rf.feature_importances_,index=fcols).sort_values(ascending=False).head(12).round(5).to_dict()
        return out

    with st.spinner("Training regression models…"):
        reg_res = train_regressors(reg_df)

    tgt_sel = st.radio("Target variable:",
                       ["Q21_WTP_AED — Willingness to Pay (AED)",
                        "Q24_DownloadLikelihood — Download Intent (1-10)"],
                       horizontal=True)
    tgt_key = "Q21_WTP_AED" if "WTP_AED" in tgt_sel else "Q24_DownloadLikelihood"
    tres = reg_res.get(tgt_key, {})

    perf_r = pd.DataFrame({mn:{"R²":f"{v['r2']:.3f}","MAE":f"{v['mae']:.2f}","RMSE":f"{v['rmse']:.2f}"}
                           for mn,v in tres.items() if isinstance(v,dict) and "r2" in v}).T.reset_index()
    perf_r.columns = ["Model","R²","MAE","RMSE"]
    st.dataframe(perf_r.reset_index(drop=True), use_container_width=True)

    best_rn = max([(k,v) for k,v in tres.items() if isinstance(v,dict) and "r2" in v],
                  key=lambda x:x[1]["r2"])[0]
    bres = tres[best_rn]
    st.markdown(f"""<div class='insight-box'>
    <b>Best: {best_rn}</b> — R²={bres['r2']:.3f} | MAE={bres['mae']:.2f} | RMSE={bres['rmse']:.2f}
    <br><i>R² of 0.25–0.35 is standard for survey-based stated-preference price models.</i>
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        yt=np.array(bres["yte"]); yp=np.array(bres["ypred"])
        fig=px.scatter(x=yt,y=yp,opacity=0.4,labels={"x":"Actual","y":"Predicted"})
        mn_v,mx_v=min(yt.min(),yp.min()),max(yt.max(),yp.max())
        fig.add_trace(go.Scatter(x=[mn_v,mx_v],y=[mn_v,mx_v],mode="lines",
                                  name="Perfect fit",line=dict(color="red",dash="dash")))
        fig.update_traces(marker=dict(color=TEAL,size=5),selector=dict(mode="markers"))
        fig.update_layout(height=360,margin=dict(t=30,b=10),title=f"R²={bres['r2']:.3f}")
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        fi_r=pd.Series(tres.get("fi",{})).sort_values().reset_index()
        fi_r.columns=["Feature","Importance"]
        fi_r["Type"]=fi_r["Feature"].apply(
            lambda x:"Engineered" if x.startswith("Eng_") else
                     "Derived" if x.startswith("Derived_") else "Survey Item")
        fig=px.bar(fi_r,x="Importance",y="Feature",orientation="h",color="Type",
                   color_discrete_map={"Engineered":TEAL,"Derived":GOLD,"Survey Item":NAVY})
        fig.update_layout(height=360,margin=dict(t=20,b=10,l=200))
        st.plotly_chart(fig,use_container_width=True)

# ─── TAB 7: BUSINESS FINDINGS ─────────────────────────────────────────────────
with tabs[6]:
    st.header("🏆 Business Findings & Strategic Recommendations")

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{NAVY},{TEAL});border-radius:12px;
                padding:24px;color:white;margin-bottom:20px;'>
      <h2 style='color:white;border:none;font-size:1.4rem;margin:0 0 10px;'>
        ✅ VERDICT: FinWise AI is Commercially Viable
      </h2>
      <p style='margin:0;font-size:.95rem;line-height:1.6;'>
      <b>{wtp_pct:.1f}%</b> of Dubai professionals surveyed are willing to pay.
      Average WTP of <b>AED {avg_wtp:.0f}/month</b> (median AED {master['Q21_WTP_AED'].median():.0f})
      supports a freemium + paid tier model. All 8 pain items score above 3.0/5 (neutral),
      confirming systemic financial management friction across Dubai's expat workforce.
      </p>
    </div>""", unsafe_allow_html=True)

    st.subheader("Top Features Respondents Want")
    feat_map = {
        "Q16_AIExpenseCategorisation":"AI Expense Categorisation",
        "Q16_MultiCurrencyTracking":"Multi-Currency Tracking",
        "Q16_SavingsGoals":"Savings Goals & Coaching",
        "Q16_RemittanceBestRate":"Best Remittance Rate",
        "Q16_InvestmentTracking":"Investment Tracking",
        "Q16_SubscriptionAlerts":"Subscription Alerts",
        "Q16_HealthDashboard":"Health Dashboard",
        "Q16_ChatbotQueries":"AI Chatbot",
        "Q16_GamifiedSavings":"Gamified Savings"
    }
    fm = master[[c for c in feat_map if c in master.columns]].mean().rename(feat_map).sort_values(ascending=False)
    fig=px.bar(fm.reset_index(),x=fm.values,y=fm.index,orientation="h",
               color=fm.values,color_continuous_scale=[SAGE,TEAL,NAVY],
               text=[f"{v:.2f}" for v in fm.values])
    fig.add_vline(x=3.5,line_dash="dot",line_color=CORAL,annotation_text="High priority (>3.5)")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=360,showlegend=False,coloraxis_showscale=False,
                       margin=dict(t=20,b=10,l=220))
    st.plotly_chart(fig,use_container_width=True)

    c1,c2,c3 = st.columns(3)
    demand_n   = int(master["Q20_WTP_Binary"].sum())
    high_dl_n  = int((master["Q24_DownloadLikelihood"]>=7).sum())
    vs_n       = int(master["Flag_ValueSeeker"].sum()) if "Flag_ValueSeeker" in master.columns else 0

    with c1:
        st.markdown(f"""<div class='metric-card'>
        <div style='color:{TEAL};font-weight:700;'>📈 Demand Funnel</div>
        <table style='width:100%;font-size:.88rem;margin-top:8px;'>
          <tr><td>Total surveyed</td><td style='text-align:right;'><b>1,200</b></td></tr>
          <tr><td>Willing to pay</td>
              <td style='text-align:right;color:{TEAL};'><b>{demand_n} ({demand_n/12:.0f}%)</b></td></tr>
          <tr><td>Download intent ≥7</td>
              <td style='text-align:right;color:{GOLD};'><b>{high_dl_n} ({high_dl_n/12:.0f}%)</b></td></tr>
          <tr><td>Value Seekers</td>
              <td style='text-align:right;color:{CORAL};'><b>{vs_n} ({vs_n/12:.0f}%)</b></td></tr>
        </table></div>""", unsafe_allow_html=True)

    with c2:
        pricing_map2 = {1:"Free w/Ads",2:"Freemium",3:"Monthly Sub",
                        4:"Annual Sub",5:"One-time",6:"Pay-per-feature"}
        top_price = master["Q22_PricingModel"].map(pricing_map2).value_counts().idxmax()
        st.markdown(f"""<div class='metric-card'>
        <div style='color:{GOLD};font-weight:700;'>💰 Pricing Recommendation</div>
        <div style='font-size:.88rem;margin-top:8px;color:#555;'>
        Most preferred model: <b>{top_price}</b><br><br>
        <b>Free tier</b> — Core tracking, AED only<br>
        <b>Plus AED 25/mo</b> — Multi-currency, 3 banks, AI<br>
        <b>Pro AED 60/mo</b> — Unlimited, investments, coaching<br><br>
        <i>Median WTP = AED {master['Q21_WTP_AED'].median():.0f} supports Plus tier.</i>
        </div></div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""<div class='metric-card'>
        <div style='color:{CORAL};font-weight:700;'>⚠️ Key Risks</div>
        <div style='font-size:.85rem;margin-top:8px;color:#555;'>
        <b>1.</b> Data privacy = #1 adoption barrier (Q25)<br><br>
        <b>2.</b> UAE regulatory approval needed (CBUAE/DFSA)<br><br>
        <b>3.</b> Competition: Wise, Revolut, Sarwa<br><br>
        <b>4.</b> Stated WTP surveys overestimate actual conversion by 4–8×
        </div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("FinWise AI · SP Jain GMBA · Data Analytics · Dubai 2026 · n=1,200 synthetic respondents")
