"""
FinWise AI — Full Analytics Engine
===================================
Runs: Descriptive → Cross-tab → Diagnostic → Classification → Clustering
      → Association Mining → Regression → Business Findings
All results serialised to outputs/ for Streamlit dashboard consumption.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings, json, os, pickle
warnings.filterwarnings("ignore")

# sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve,
                              f1_score, precision_score, recall_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Association
from mlxtend.frequent_patterns import apriori, association_rules

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data/"
OUT_DIR  = "outputs/"
FIG_DIR  = "outputs/figures/"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
NAVY  = "#1B3A6B"
TEAL  = "#0A7C8C"
GOLD  = "#C8A84B"
CORAL = "#E07B5A"
SAGE  = "#5A8A6B"
LGRAY = "#F2F4F8"
PERSONA_COLORS = {
    "Tech-Savvy Millennial": TEAL,
    "Struggling Expat":      CORAL,
    "Senior Finance Pro":    NAVY,
    "High-Earning Nomad":    GOLD,
    "Cautious Saver":        SAGE,
}
PALETTE = [NAVY, TEAL, GOLD, CORAL, SAGE, "#8B5CF6", "#EC4899"]

def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}{name}.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight='bold', color=NAVY, pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color='#444')
    ax.set_ylabel(ylabel, fontsize=10, color='#444')
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=9)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading datasets...")
master  = pd.read_csv(f"{DATA_DIR}FinWise_Master_Cleaned.csv")
clf_df  = pd.read_csv(f"{DATA_DIR}FinWise_Classification_Dataset.csv")
clust_df= pd.read_csv(f"{DATA_DIR}FinWise_Clustering_Dataset.csv")
assoc_df= pd.read_csv(f"{DATA_DIR}FinWise_Association_Dataset.csv")
reg_df  = pd.read_csv(f"{DATA_DIR}FinWise_Regression_Dataset.csv")
print(f"  Master: {master.shape} | Clf: {clf_df.shape} | Clust: {clust_df.shape}")
print(f"  Assoc:  {assoc_df.shape} | Reg: {reg_df.shape}")

results = {}  # will be serialised at end

# ══════════════════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Descriptive Analytics...")

desc = {}

# 1a WTP demand signal
wtp_dist = master['Q20_WTP_Binary'].value_counts()
desc['wtp_willing_pct']   = round(master['Q20_WTP_Binary'].mean()*100, 1)
desc['wtp_dist']          = wtp_dist.to_dict()
desc['avg_download_score']= round(master['Q24_DownloadLikelihood'].mean(), 2)
desc['avg_nps']           = round(master['Q29_NPS'].mean(), 2)
desc['nps_breakdown']     = master['Q29_NPS_Category'].value_counts().to_dict()
desc['avg_pain']          = round(master['Derived_PainComposite'].mean(), 2)
desc['avg_wtp_aed']       = round(master['Q21_WTP_AED'].mean(), 1)
desc['median_wtp_aed']    = round(master['Q21_WTP_AED'].median(), 1)

# Fig 1 — Demand signal overview (4-panel)
fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor='white')
fig.suptitle('FinWise AI — Demand Signal Overview', fontsize=15, fontweight='bold', color=NAVY, y=1.01)

# WTP binary pie
ax = axes[0,0]
sizes = [wtp_dist.get(1,0), wtp_dist.get(0,0)]
labels = [f'Willing to Pay\n{sizes[0]} ({sizes[0]/1200*100:.1f}%)',
          f'Not Willing\n{sizes[1]} ({sizes[1]/1200*100:.1f}%)']
wedges, texts = ax.pie(sizes, labels=labels, colors=[TEAL, CORAL],
                        startangle=90, wedgeprops=dict(width=0.6), textprops={'fontsize':9})
ax.set_title('Willingness to Pay\n(Q20 — Binary)', fontsize=12, fontweight='bold', color=NAVY)

# Download likelihood distribution
ax = axes[0,1]
dl_counts = master['Q24_DownloadLikelihood'].value_counts().sort_index()
bars = ax.bar(dl_counts.index, dl_counts.values, color=[TEAL if v >= 7 else GOLD if v >= 5 else CORAL for v in dl_counts.index], edgecolor='white', linewidth=0.5)
ax.axvline(master['Q24_DownloadLikelihood'].mean(), color=NAVY, linestyle='--', linewidth=1.5, label=f"Mean = {master['Q24_DownloadLikelihood'].mean():.1f}")
ax.legend(fontsize=8)
style_ax(ax, 'Download Likelihood Distribution\n(Q24 — 1 to 10 scale)', 'Score', 'Count')

# NPS breakdown
ax = axes[1,0]
nps_data = master['Q29_NPS_Category'].value_counts()
nps_colors = {'Promoter': TEAL, 'Passive': GOLD, 'Detractor': CORAL}
bars = ax.bar(nps_data.index, nps_data.values,
              color=[nps_colors.get(k, NAVY) for k in nps_data.index], edgecolor='white')
for bar, val in zip(bars, nps_data.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, str(val),
            ha='center', va='bottom', fontsize=9, fontweight='bold')
style_ax(ax, 'NPS Category Breakdown\n(Q29 — Net Promoter Score)', 'Category', 'Count')

# WTP AED histogram
ax = axes[1,1]
ax.hist(master['Q21_WTP_AED'], bins=20, color=TEAL, edgecolor='white', alpha=0.85)
ax.axvline(master['Q21_WTP_AED'].mean(), color=CORAL, linestyle='--', lw=2,
           label=f"Mean = AED {master['Q21_WTP_AED'].mean():.0f}")
ax.axvline(master['Q21_WTP_AED'].median(), color=GOLD, linestyle='--', lw=2,
           label=f"Median = AED {master['Q21_WTP_AED'].median():.0f}")
ax.legend(fontsize=8)
style_ax(ax, 'Willingness to Pay Distribution\n(Q21 — AED/month)', 'AED/month', 'Count')

save_fig("01_demand_signal_overview")
print("   Fig 01 saved")

# Fig 2 — Demographics
fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor='white')
fig.suptitle('FinWise AI — Respondent Demographics', fontsize=15, fontweight='bold', color=NAVY, y=1.01)

demo_fields = [
    ('Q1_AgeGroup', {1:'<25',2:'25-34',3:'35-44',4:'45-54',5:'55+'}, 'Age Group'),
    ('Q2_Gender',   {1:'Male',2:'Female',3:'Non-binary',4:'PNS'}, 'Gender'),
    ('Q3_Education',{1:'High School',2:"Bachelor's",3:'Master/MBA',4:'PhD',5:'Prof.Qual'}, 'Education'),
    ('Q4_Employment',{1:'Private',2:'Govt',3:'Freelance',4:'Business',5:'Part-time'}, 'Employment'),
    ('Q5_IncomeTier',{1:'<5k',2:'5-10k',3:'10-20k',4:'20-40k',5:'>40k',6:'PNS'}, 'Income (AED)'),
    ('Q7_YearsDubai',{1:'<1yr',2:'1-3yr',3:'3-5yr',4:'5-10yr',5:'>10yr'}, 'Years in Dubai'),
]
for idx, (col, mapping, label) in enumerate(demo_fields):
    ax = axes[idx//3][idx%3]
    data = master[col].map(mapping).value_counts()
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(data)))[::-1]
    bars = ax.bar(range(len(data)), data.values, color=colors, edgecolor='white')
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data.index, rotation=30, ha='right', fontsize=8)
    for bar, val in zip(bars, data.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, str(val),
                ha='center', va='bottom', fontsize=7, fontweight='bold', color=NAVY)
    style_ax(ax, label, '', 'Count')

save_fig("02_demographics")
print("   Fig 02 saved")

# Fig 3 — Pain points heatmap
pain_cols = [c for c in master.columns if c.startswith('Q13_')]
pain_labels = {
    'Q13_DiffTrackMultiCurrency': 'Multi-currency tracking',
    'Q13_LoseFXMoney':            'FX money loss',
    'Q13_SavingsHabit':           'Savings habit',
    'Q13_MultiBankTimeConsuming': 'Multi-bank mgmt',
    'Q13_DissatisifiedTools':     'Tool dissatisfaction',
    'Q13_UnexpectedExpenses':     'Unexpected expenses',
    'Q13_HealthUnderstanding':    'Financial health clarity',
    'Q13_Overspent':              'Overspending incidents'
}
pain_means = master[pain_cols].mean().rename(pain_labels)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
fig.suptitle('Pain Point Analysis — Q13 Likert Scores (1-5)', fontsize=14, fontweight='bold', color=NAVY)

# Horizontal bar chart
ax = axes[0]
colors = [CORAL if v >= 3.5 else GOLD if v >= 3.0 else TEAL for v in pain_means.values]
bars = ax.barh(range(len(pain_means)), pain_means.values, color=colors, edgecolor='white')
ax.set_yticks(range(len(pain_means)))
ax.set_yticklabels(pain_means.index, fontsize=9)
ax.axvline(3.0, color='gray', linestyle='--', lw=1, alpha=0.5, label='Neutral (3.0)')
for bar, val in zip(bars, pain_means.values):
    ax.text(val+0.02, bar.get_y()+bar.get_height()/2, f'{val:.2f}',
            va='center', fontsize=8, fontweight='bold')
ax.legend(fontsize=8)
style_ax(ax, 'Mean Pain Score by Item', 'Mean Score (1-5)', '')
ax.set_xlim(0, 5.2)

# Pain by persona heatmap
pain_persona = master.groupby('Persona_Cluster')[pain_cols].mean().rename(columns=pain_labels)
ax = axes[1]
sns.heatmap(pain_persona, annot=True, fmt='.1f', cmap='RdYlGn_r',
            vmin=1, vmax=5, ax=ax, cbar_kws={'shrink':0.8}, annot_kws={'size':8})
ax.set_title('Pain Scores by Persona', fontsize=12, fontweight='bold', color=NAVY)
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=40, labelsize=8)
ax.tick_params(axis='y', rotation=0, labelsize=9)

save_fig("03_pain_points")
print("   Fig 03 saved")

results['descriptive'] = desc

# ══════════════════════════════════════════════════════════════════════════════
# 2. CROSS-TABULATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Cross-Tabulation...")

# WTP by Income tier
wtp_income = pd.crosstab(
    master['Q5_IncomeTier'].map({1:'<5k',2:'5-10k',3:'10-20k',4:'20-40k',5:'>40k',6:'PNS'}),
    master['Q20_WTP_Binary'].map({0:'Not Willing',1:'Willing'}),
    normalize='index'
) * 100

# WTP by Age group
wtp_age = pd.crosstab(
    master['Q1_AgeGroup'].map({1:'<25',2:'25-34',3:'35-44',4:'45-54',5:'55+'}),
    master['Q20_WTP_Binary'].map({0:'Not Willing',1:'Willing'}),
    normalize='index'
) * 100

# WTP by Persona
wtp_persona = pd.crosstab(
    master['Persona_Cluster'],
    master['Q20_WTP_Binary'].map({0:'Not Willing',1:'Willing'}),
    normalize='index'
) * 100

fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor='white')
fig.suptitle('Cross-Tabulation — WTP by Segment', fontsize=14, fontweight='bold', color=NAVY)

for ax, data, title in zip(axes,
    [wtp_income, wtp_age, wtp_persona],
    ['WTP % by Income Tier', 'WTP % by Age Group', 'WTP % by Persona']):
    if 'Willing' in data.columns:
        bars = ax.barh(range(len(data)), data['Willing'], color=TEAL, edgecolor='white', label='Willing')
        ax.barh(range(len(data)), data['Not Willing'], left=data['Willing'],
                color=CORAL, edgecolor='white', label='Not Willing')
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data.index, fontsize=8)
        for bar, val in zip(bars, data['Willing']):
            ax.text(val/2, bar.get_y()+bar.get_height()/2, f'{val:.0f}%',
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlim(0, 100)
        style_ax(ax, title, '% of segment', '')

save_fig("04_crosstab_wtp")
print("   Fig 04 saved")

# Store crosstab summaries
results['crosstab'] = {
    'wtp_by_persona': wtp_persona.round(1).to_dict(),
    'wtp_by_income':  wtp_income.round(1).to_dict(),
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. DIAGNOSTIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Diagnostic Analysis...")

# Correlation heatmap of key variables
diag_cols = ['Derived_PainComposite','Derived_TechAffinity','Derived_AdoptionAttitude',
             'Derived_FinancialStress','Q5_IncomeTier','Q10_NumCurrencies',
             'Q24_DownloadLikelihood','Q29_NPS','Q21_WTP_AED','Q20_WTP_Binary',
             'Eng_PainGap','Eng_AdoptionReadiness']
corr_mat = master[diag_cols].corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
fig.suptitle('Diagnostic Analysis — Correlation & Distribution', fontsize=14, fontweight='bold', color=NAVY)

ax = axes[0]
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={'shrink':0.8},
            annot_kws={'size':7}, linewidths=0.3)
ax.set_title('Key Variable Correlation Matrix', fontsize=12, fontweight='bold', color=NAVY)
ax.tick_params(axis='both', labelsize=7)

# Pain vs WTP scatter by persona
ax = axes[1]
for persona, color in PERSONA_COLORS.items():
    mask_p = master['Persona_Cluster'] == persona
    ax.scatter(master.loc[mask_p, 'Derived_PainComposite'],
               master.loc[mask_p, 'Q21_WTP_AED'],
               c=color, label=persona, alpha=0.4, s=18, edgecolors='none')
# Regression line
m, b, r, p, _ = stats.linregress(master['Derived_PainComposite'], master['Q21_WTP_AED'])
x_line = np.linspace(master['Derived_PainComposite'].min(), master['Derived_PainComposite'].max(), 100)
ax.plot(x_line, m*x_line + b, color=NAVY, linewidth=2, linestyle='--', label=f'Trend (r={r:.2f})')
ax.legend(fontsize=7, markerscale=1.5)
style_ax(ax, 'Pain Composite vs WTP — Coloured by Persona',
         'Pain Composite Score', 'WTP (AED/month)')

save_fig("05_diagnostic_correlation")
print("   Fig 05 saved")

# Fig — Feature importance preview (mutual info proxy via correlation)
pain_wtp_corr = master[['Q13_DiffTrackMultiCurrency','Q13_LoseFXMoney','Q13_SavingsHabit',
    'Q13_MultiBankTimeConsuming','Q13_DissatisifiedTools','Q13_UnexpectedExpenses',
    'Q13_HealthUnderstanding','Q13_Overspent',
    'Q16_AIExpenseCategorisation','Q16_MultiCurrencyTracking','Q16_SavingsGoals',
    'Q16_RemittanceBestRate','Q21_WTP_AED']].corr()['Q21_WTP_AED'].drop('Q21_WTP_AED')

fig, ax = plt.subplots(figsize=(10,5), facecolor='white')
colors = [TEAL if v > 0 else CORAL for v in pain_wtp_corr.values]
bars = ax.barh(range(len(pain_wtp_corr)), pain_wtp_corr.values, color=colors, edgecolor='white')
ax.set_yticks(range(len(pain_wtp_corr)))
ax.set_yticklabels([c.replace('Q13_','').replace('Q16_','') for c in pain_wtp_corr.index], fontsize=9)
ax.axvline(0, color='gray', lw=0.8)
for bar, val in zip(bars, pain_wtp_corr.values):
    ax.text(val + (0.005 if val >= 0 else -0.005), bar.get_y()+bar.get_height()/2,
            f'{val:.2f}', va='center', ha='left' if val >= 0 else 'right', fontsize=8)
style_ax(ax, 'Pain & Feature Preference Correlations with WTP (AED)',
         'Pearson r', 'Survey Item')
save_fig("06_diagnostic_feature_corr")
print("   Fig 06 saved")

results['diagnostic'] = {
    'pain_wtp_r': round(float(stats.pearsonr(master['Derived_PainComposite'], master['Q21_WTP_AED'])[0]), 3),
    'pain_dl_r':  round(float(stats.pearsonr(master['Derived_PainComposite'], master['Q24_DownloadLikelihood'])[0]), 3),
    'tech_wtp_r': round(float(stats.pearsonr(master['Derived_TechAffinity'],   master['Q21_WTP_AED'])[0]), 3),
}

# ══════════════════════════════════════════════════════════════════════════════
# 4. CLASSIFICATION MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Classification Models...")

# Feature prep - use key features (avoid leakage columns)
LEAKAGE = ['Q20_WTP_Binary','Q20_WTP_Intent','Q21_WTP_AED','Q21_WTP_Tier',
           'Eng_WTPperPain','Respondent_ID','Persona_Cluster',
           'Persona_Cluster_Enc','Q29_NPS_Category']

clf_features = [c for c in clf_df.columns
                if c not in LEAKAGE
                and not c.endswith('_Label')
                and clf_df[c].dtype in [np.float64, np.int64]]

X = clf_df[clf_features].fillna(0)
y = clf_df['Q20_WTP_Binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    'KNN':               KNeighborsClassifier(n_neighbors=9, metric='euclidean'),
    'Decision Tree':     DecisionTreeClassifier(max_depth=7, min_samples_split=20,
                                                min_samples_leaf=10, random_state=42),
    'Random Forest':     RandomForestClassifier(n_estimators=200, max_depth=10,
                                                min_samples_leaf=5, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                     learning_rate=0.08, random_state=42),
}

clf_results = {}
for name, model in models.items():
    use_scaled = name == 'KNN'
    Xtr = X_train_sc if use_scaled else X_train
    Xte = X_test_sc  if use_scaled else X_test

    model.fit(Xtr, y_train)

    y_pred_train = model.predict(Xtr)
    y_pred_test  = model.predict(Xte)
    y_prob       = model.predict_proba(Xte)[:, 1] if hasattr(model, 'predict_proba') else None

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test,  y_pred_test)
    prec      = precision_score(y_test, y_pred_test)
    rec       = recall_score(y_test,    y_pred_test)
    f1        = f1_score(y_test,        y_pred_test)
    auc       = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    cm        = confusion_matrix(y_test, y_pred_test)

    clf_results[name] = {
        'train_acc': round(train_acc, 4),
        'test_acc':  round(test_acc, 4),
        'precision': round(prec, 4),
        'recall':    round(rec, 4),
        'f1':        round(f1, 4),
        'auc':       round(auc, 4) if auc else None,
        'cm':        cm.tolist(),
    }
    auc_str = f"{auc:.3f}" if auc is not None else "N/A"
    print(f"   {name:<22}: Test={test_acc:.3f} | F1={f1:.3f} | AUC={auc_str}")

# Feature importance (RF)
rf_model = models['Random Forest']
fi = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
clf_results['rf_feature_importance'] = fi.round(4).to_dict()

# Save scaler and best model
with open(f"{OUT_DIR}scaler_clf.pkl", 'wb') as f:
    pickle.dump(scaler, f)
with open(f"{OUT_DIR}rf_model.pkl", 'wb') as f:
    pickle.dump(rf_model, f)
with open(f"{OUT_DIR}clf_features.pkl", 'wb') as f:
    pickle.dump(clf_features, f)

# ── Plot classification results ────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 11), facecolor='white')
fig.suptitle('Classification Results — WTP Binary Prediction', fontsize=14, fontweight='bold', color=NAVY)

# Model comparison bar
ax = axes[0,0]
metrics_df = pd.DataFrame({
    name: {k: v for k,v in v.items() if k in ['train_acc','test_acc','f1','precision','recall','auc']}
    for name, v in clf_results.items() if isinstance(v, dict) and 'test_acc' in v
}).T
x = np.arange(len(metrics_df))
w = 0.15
for i, (col, color) in enumerate(zip(['train_acc','test_acc','f1','auc'], [NAVY, TEAL, GOLD, CORAL])):
    ax.bar(x + i*w, metrics_df[col].astype(float), w, label=col.replace('_',' ').title(), color=color, edgecolor='white')
ax.set_xticks(x + 1.5*w)
ax.set_xticklabels(metrics_df.index, fontsize=8, rotation=15)
ax.legend(fontsize=7)
ax.set_ylim(0.5, 1.05)
style_ax(ax, 'Model Performance Comparison', '', 'Score')

# Confusion matrices (3 key models)
for idx, mname in enumerate(['Decision Tree','Random Forest','Gradient Boosting']):
    ax = axes[0 if idx < 2 else 1][idx%2 + (1 if idx < 2 else 0)]
    cm = np.array(clf_results[mname]['cm'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Willing','Willing'],
                yticklabels=['Not Willing','Willing'],
                cbar=False, annot_kws={'size':12})
    ax.set_title(f'Confusion Matrix\n{mname}', fontsize=10, fontweight='bold', color=NAVY)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)

# ROC curves
ax = axes[1,1]
for name, model in models.items():
    use_scaled = name == 'KNN'
    Xte_use = X_test_sc if use_scaled else X_test
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(Xte_use)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = clf_results[name]['auc']
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.2f})', linewidth=1.5)
ax.plot([0,1],[0,1], 'k--', lw=0.8, alpha=0.5)
ax.legend(fontsize=8)
style_ax(ax, 'ROC Curves — All Models', 'False Positive Rate', 'True Positive Rate')
ax.set_xlim(0,1); ax.set_ylim(0,1.02)

# Feature importance RF
ax = axes[1,2]
top15 = pd.Series(clf_results['rf_feature_importance']).sort_values()
colors = [TEAL if 'Eng_' in k else GOLD if 'Derived_' in k else NAVY for k in top15.index]
ax.barh(range(len(top15)), top15.values, color=colors, edgecolor='white')
ax.set_yticks(range(len(top15)))
ax.set_yticklabels([k.replace('Q13_','').replace('Q16_','').replace('Eng_','')
                     .replace('Derived_','').replace('Q26_','').replace('Q24_','')
                     for k in top15.index], fontsize=8)
style_ax(ax, 'Top 15 Feature Importances\n(Random Forest)', 'Importance', '')
legend_patches = [mpatches.Patch(color=TEAL, label='Engineered'),
                  mpatches.Patch(color=GOLD, label='Derived Composite'),
                  mpatches.Patch(color=NAVY, label='Survey Item')]
ax.legend(handles=legend_patches, fontsize=7, loc='lower right')

save_fig("07_classification_results")
print("   Fig 07 saved")

results['classification'] = clf_results

# ══════════════════════════════════════════════════════════════════════════════
# 5. CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Clustering...")

clust_numeric_cols = [c for c in clust_df.select_dtypes(include=[np.number]).columns
                      if 'Persona_Cluster_Enc' not in c]
Xc = clust_df[clust_numeric_cols].fillna(0)
sc_clust = StandardScaler()
Xc_sc = sc_clust.fit_transform(Xc)

# Elbow + Silhouette to confirm k=5
k_range = range(2, 9)
inertias, sil_scores = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(Xc_sc)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(Xc_sc, km.labels_, sample_size=500, random_state=42))

# Fig — Elbow & Silhouette
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
fig.suptitle('K-Means — Optimal Cluster Selection', fontsize=14, fontweight='bold', color=NAVY)
ax = axes[0]
ax.plot(list(k_range), inertias, 'o-', color=TEAL, linewidth=2, markersize=7)
ax.axvline(5, color=CORAL, linestyle='--', lw=1.5, label='k=5 selected')
ax.legend(fontsize=9)
style_ax(ax, 'Elbow Curve (Inertia vs k)', 'Number of Clusters k', 'Inertia (WCSS)')
ax = axes[1]
ax.plot(list(k_range), sil_scores, 's-', color=GOLD, linewidth=2, markersize=7)
ax.axvline(5, color=CORAL, linestyle='--', lw=1.5, label='k=5 selected')
best_k = list(k_range)[np.argmax(sil_scores)]
ax.axvline(best_k, color=SAGE, linestyle=':', lw=1.5, label=f'Best sil. k={best_k}')
ax.legend(fontsize=9)
style_ax(ax, 'Silhouette Score vs k', 'Number of Clusters k', 'Silhouette Score')
save_fig("08_cluster_elbow")
print("   Fig 08 saved")

# Fit k=5
km5 = KMeans(n_clusters=5, random_state=42, n_init=20)
km5.fit(Xc_sc)
clust_df['KMeans_Label'] = km5.labels_
sil5  = silhouette_score(Xc_sc, km5.labels_, sample_size=600, random_state=42)
db5   = davies_bouldin_score(Xc_sc, km5.labels_)
print(f"   k=5 Silhouette={sil5:.3f} | DB={db5:.3f}")

# PCA 2D visualisation
pca = PCA(n_components=2, random_state=42)
Xc_pca = pca.fit_transform(Xc_sc)
var_exp = pca.explained_variance_ratio_

CLUSTER_NAMES = {
    0: "Budget-Conscious\nExpats",
    1: "Cautious\nSavers",
    2: "Digital-First\nMillennials",
    3: "High-Value\nNomads",
    4: "Finance\nProfessionals"
}
CLUSTER_COLORS = [CORAL, SAGE, TEAL, GOLD, NAVY]

# Map cluster IDs to business names by profiling them
clust_profile = clust_df.groupby('KMeans_Label')[
    ['Derived_PainComposite','Derived_TechAffinity','Q21_WTP_AED',
     'Q1_AgeRaw','Q5_IncomeTier_Imputed','Eng_AdoptionReadiness']
].mean().round(2)

# Assign names by ranking on pain (highest pain = Budget Expat, lowest = Finance Pro, etc.)
pain_rank     = clust_profile['Derived_PainComposite'].rank().astype(int)
tech_rank     = clust_profile['Derived_TechAffinity'].rank().astype(int)
income_rank   = clust_profile['Q5_IncomeTier_Imputed'].rank().astype(int)
age_rank      = clust_profile['Q1_AgeRaw'].rank().astype(int)

cluster_name_map = {}
for cid in range(5):
    if pain_rank[cid] == 5 and income_rank[cid] <= 2:
        cluster_name_map[cid] = "Budget-Conscious Expats"
    elif income_rank[cid] >= 4 and tech_rank[cid] >= 3:
        cluster_name_map[cid] = "High-Value Nomads"
    elif tech_rank[cid] == 5 and age_rank[cid] <= 2:
        cluster_name_map[cid] = "Digital-First Millennials"
    elif pain_rank[cid] <= 2 and income_rank[cid] >= 3:
        cluster_name_map[cid] = "Finance Professionals"
    else:
        cluster_name_map[cid] = "Cautious Savers"

clust_df['Cluster_Name'] = clust_df['KMeans_Label'].map(cluster_name_map)

# Build profile table
cluster_profile_full = clust_df.groupby('Cluster_Name').agg(
    n=('KMeans_Label','count'),
    pain=('Derived_PainComposite','mean'),
    tech=('Derived_TechAffinity','mean'),
    wtp=('Q21_WTP_AED','mean'),
    readiness=('Eng_AdoptionReadiness','mean'),
    age=('Q1_AgeRaw','mean'),
    income=('Q5_IncomeTier_Imputed','mean')
).round(2)

print("   Cluster profiles:")
print(cluster_profile_full)

# Fig — Cluster visualisation (3-panel)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
fig.suptitle('K-Means Clustering — Customer Segment Analysis', fontsize=14, fontweight='bold', color=NAVY)

# PCA scatter
ax = axes[0]
unique_clusters = sorted(clust_df['Cluster_Name'].unique())
cmap = {name: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, name in enumerate(unique_clusters)}
for cname in unique_clusters:
    mask = clust_df['Cluster_Name'] == cname
    ax.scatter(Xc_pca[mask, 0], Xc_pca[mask, 1],
               c=cmap[cname], label=cname, alpha=0.5, s=15, edgecolors='none')
ax.legend(fontsize=7, markerscale=2, loc='upper right')
style_ax(ax, f'PCA 2D Projection\n({var_exp[0]*100:.0f}% + {var_exp[1]*100:.0f}% variance)',
         'PC1', 'PC2')

# Radar chart (spider) — cluster profiles
ax = axes[1]
dims = ['Pain', 'Tech\nAffinity', 'WTP\n(norm)', 'Age\n(norm)', 'Readiness']
profile_norm = clust_df.groupby('Cluster_Name').agg({
    'Derived_PainComposite': 'mean',
    'Derived_TechAffinity':  'mean',
    'Q21_WTP_AED':           'mean',
    'Q1_AgeRaw':             'mean',
    'Eng_AdoptionReadiness': 'mean'
})
# Normalise 0-1
for col in profile_norm.columns:
    profile_norm[col] = (profile_norm[col] - profile_norm[col].min()) / (profile_norm[col].max() - profile_norm[col].min() + 1e-9)

for i, (cname, row) in enumerate(profile_norm.iterrows()):
    vals = list(row.values)
    xpos = np.arange(len(vals))
    ax.plot(xpos, vals, 'o-', color=cmap[cname], label=cname, linewidth=1.5, markersize=5, alpha=0.8)
    ax.fill_between(xpos, vals, alpha=0.05, color=cmap[cname])
ax.set_xticks(range(len(dims)))
ax.set_xticklabels(dims, fontsize=8)
ax.legend(fontsize=7)
style_ax(ax, 'Normalised Cluster Profiles', '', 'Score (0-1 normalised)')

# WTP by cluster bar
ax = axes[2]
wtp_by_cluster = clust_df.groupby('Cluster_Name')['Q21_WTP_AED'].mean().sort_values(ascending=False)
bars = ax.bar(range(len(wtp_by_cluster)), wtp_by_cluster.values,
              color=[cmap[n] for n in wtp_by_cluster.index], edgecolor='white')
ax.set_xticks(range(len(wtp_by_cluster)))
ax.set_xticklabels(wtp_by_cluster.index, rotation=20, ha='right', fontsize=8)
for bar, val in zip(bars, wtp_by_cluster.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'AED {val:.0f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
style_ax(ax, 'Average WTP by Cluster\n(AED/month)', '', 'Mean WTP (AED)')

save_fig("09_clustering_results")
print("   Fig 09 saved")

results['clustering'] = {
    'silhouette': round(sil5, 4),
    'db_score':   round(db5, 4),
    'variance_explained_2d': [round(float(v), 4) for v in var_exp],
    'cluster_profile': cluster_profile_full.to_dict(),
    'cluster_name_map': cluster_name_map,
}

# Save clustering model
with open(f"{OUT_DIR}kmeans_model.pkl", 'wb') as f:
    pickle.dump(km5, f)
with open(f"{OUT_DIR}pca_model.pkl", 'wb') as f:
    pickle.dump(pca, f)
with open(f"{OUT_DIR}scaler_clust.pkl", 'wb') as f:
    pickle.dump(sc_clust, f)
clust_df.to_csv(f"{OUT_DIR}FinWise_Clustered.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 6. ASSOCIATION RULE MINING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Association Rule Mining...")

basket = assoc_df.astype(bool)

freq_items = apriori(basket, min_support=0.10, use_colnames=True, max_len=4)
rules = association_rules(freq_items, metric='lift', min_threshold=1.05)
rules = rules.sort_values('lift', ascending=False)

# Filter meaningful rules
rules_filtered = rules[
    (rules['confidence'] >= 0.55) &
    (rules['lift'] >= 1.1) &
    (rules['support'] >= 0.10)
].copy()

rules_filtered['antecedents_str'] = rules_filtered['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
rules_filtered['consequents_str'] = rules_filtered['consequents'].apply(lambda x: ', '.join(sorted(list(x))))

print(f"   Total rules: {len(rules)} | Filtered (conf≥0.55, lift≥1.1): {len(rules_filtered)}")

# Top 20 rules
top_rules = rules_filtered.nlargest(20, 'lift')[
    ['antecedents_str','consequents_str','support','confidence','lift']
].reset_index(drop=True)
top_rules.to_csv(f"{OUT_DIR}association_rules.csv", index=False)

# Fig — Association rules
fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor='white')
fig.suptitle('Association Rule Mining — Behaviour & Adoption Patterns', fontsize=14, fontweight='bold', color=NAVY)

# Support vs Confidence scatter coloured by lift
ax = axes[0]
sc = ax.scatter(rules_filtered['support'], rules_filtered['confidence'],
                c=rules_filtered['lift'], cmap='RdYlGn', s=30, alpha=0.7, edgecolors='none')
plt.colorbar(sc, ax=ax, label='Lift')
ax.axhline(0.65, color='gray', linestyle='--', lw=1, alpha=0.5)
ax.axvline(0.15, color='gray', linestyle='--', lw=1, alpha=0.5)
style_ax(ax, 'Rules — Support vs Confidence\n(colour = Lift)', 'Support', 'Confidence')

# Top 15 rules by lift
ax = axes[1]
top15_rules = top_rules.head(15)
y_labels = [f"{row['antecedents_str'][:25]} → {row['consequents_str'][:20]}"
            for _, row in top15_rules.iterrows()]
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top15_rules)))[::-1]
bars = ax.barh(range(len(top15_rules)), top15_rules['lift'], color=colors, edgecolor='white')
ax.set_yticks(range(len(top15_rules)))
ax.set_yticklabels(y_labels, fontsize=7)
ax.axvline(1.0, color='gray', linestyle='--', lw=0.8)
for bar, val in zip(bars, top15_rules['lift']):
    ax.text(val+0.005, bar.get_y()+bar.get_height()/2, f'{val:.2f}',
            va='center', fontsize=7)
style_ax(ax, 'Top 15 Rules by Lift', 'Lift', '')

save_fig("10_association_rules")
print("   Fig 10 saved")

results['association'] = {
    'total_rules':    len(rules),
    'filtered_rules': len(rules_filtered),
    'top_rules':      top_rules.head(10).to_dict('records'),
}

# ══════════════════════════════════════════════════════════════════════════════
# 7. REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Regression Models...")

# Feature prep — drop leakage for WTP regression
LEAKAGE_REG = ['Q21_WTP_AED','Q24_DownloadLikelihood','Respondent_ID',
               'Eng_WTPperPain','Eng_AdoptionReadiness',
               'Q20_WTP_Binary','Q20_WTP_Intent','Q21_WTP_Tier',
               'Persona_Cluster','Persona_Cluster_Enc','Q29_NPS_Category']

reg_feat_cols = [c for c in reg_df.select_dtypes(include=[np.number]).columns
                 if c not in LEAKAGE_REG and not c.endswith('_Label')]

Xr = reg_df[reg_feat_cols].fillna(0)
y_wtp = reg_df['Q21_WTP_AED']
y_dl  = reg_df['Q24_DownloadLikelihood']

sc_reg = StandardScaler()
Xr_sc  = sc_reg.fit_transform(Xr)

Xr_tr, Xr_te, y_wtp_tr, y_wtp_te = train_test_split(Xr_sc, y_wtp, test_size=0.2, random_state=42)
_,     _,     y_dl_tr,  y_dl_te  = train_test_split(Xr_sc, y_dl,  test_size=0.2, random_state=42)

reg_models = {
    'Linear Regression':  LinearRegression(),
    'Ridge':              Ridge(alpha=1.0),
    'Random Forest Reg':  RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
    'Gradient Boosting Reg': GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42),
}

reg_results = {'wtp': {}, 'download': {}}

for target_name, y_tr, y_te in [('wtp', y_wtp_tr, y_wtp_te), ('download', y_dl_tr, y_dl_te)]:
    for name, model in reg_models.items():
        model.fit(Xr_tr, y_tr)
        y_pred = model.predict(Xr_te)
        r2   = r2_score(y_te, y_pred)
        mae  = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        reg_results[target_name][name] = {
            'R2': round(r2, 4), 'MAE': round(mae, 4), 'RMSE': round(rmse, 4)
        }
        print(f"   [{target_name}] {name:<25}: R²={r2:.3f} | MAE={mae:.2f} | RMSE={rmse:.2f}")

# Feature importance for RF regressor (WTP)
rf_reg = reg_models['Random Forest Reg']
fi_reg = pd.Series(rf_reg.feature_importances_, index=reg_feat_cols).sort_values(ascending=False).head(15)
reg_results['rf_fi_wtp'] = fi_reg.round(5).to_dict()

# Save
with open(f"{OUT_DIR}scaler_reg.pkl", 'wb') as f:
    pickle.dump(sc_reg, f)
with open(f"{OUT_DIR}rf_reg_model.pkl", 'wb') as f:
    pickle.dump(rf_reg, f)
with open(f"{OUT_DIR}reg_feat_cols.pkl", 'wb') as f:
    pickle.dump(reg_feat_cols, f)

# Re-predict for actual vs predicted chart
best_reg = reg_models['Gradient Boosting Reg']
best_reg.fit(Xr_tr, y_wtp_tr)
y_wtp_pred = best_reg.predict(Xr_te)

# Fig — Regression results
fig, axes = plt.subplots(2, 3, figsize=(16, 11), facecolor='white')
fig.suptitle('Regression Analysis — Predicting WTP & Download Likelihood', fontsize=14, fontweight='bold', color=NAVY)

# WTP model comparison
ax = axes[0,0]
wtp_r2 = {k: v['R2'] for k,v in reg_results['wtp'].items()}
bar_colors = [TEAL if v == max(wtp_r2.values()) else NAVY for v in wtp_r2.values()]
bars = ax.bar(range(len(wtp_r2)), wtp_r2.values(), color=bar_colors, edgecolor='white')
ax.set_xticks(range(len(wtp_r2)))
ax.set_xticklabels([k.replace(' Reg','').replace(' Regression','') for k in wtp_r2.keys()],
                    rotation=20, ha='right', fontsize=8)
for bar, val in zip(bars, wtp_r2.values()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.3f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
style_ax(ax, 'WTP Regression — R² Comparison', '', 'R² Score')
ax.set_ylim(0, max(wtp_r2.values())*1.2)

# DL model comparison
ax = axes[0,1]
dl_r2 = {k: v['R2'] for k,v in reg_results['download'].items()}
bar_colors = [GOLD if v == max(dl_r2.values()) else SAGE for v in dl_r2.values()]
bars = ax.bar(range(len(dl_r2)), dl_r2.values(), color=bar_colors, edgecolor='white')
ax.set_xticks(range(len(dl_r2)))
ax.set_xticklabels([k.replace(' Reg','').replace(' Regression','') for k in dl_r2.keys()],
                    rotation=20, ha='right', fontsize=8)
for bar, val in zip(bars, dl_r2.values()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f'{val:.3f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
style_ax(ax, 'Download Likelihood — R² Comparison', '', 'R² Score')
ax.set_ylim(0, max(dl_r2.values())*1.2)

# Actual vs predicted WTP (GB)
ax = axes[0,2]
ax.scatter(y_wtp_te, y_wtp_pred, alpha=0.4, s=15, color=TEAL, edgecolors='none')
line_range = [min(y_wtp_te.min(), y_wtp_pred.min()), max(y_wtp_te.max(), y_wtp_pred.max())]
ax.plot(line_range, line_range, 'r--', lw=1.5, label='Perfect fit')
r2_best = r2_score(y_wtp_te, y_wtp_pred)
ax.text(0.05, 0.92, f'R² = {r2_best:.3f}', transform=ax.transAxes,
        fontsize=10, fontweight='bold', color=NAVY)
ax.legend(fontsize=8)
style_ax(ax, 'Actual vs Predicted WTP\n(Gradient Boosting)', 'Actual WTP (AED)', 'Predicted WTP (AED)')

# Residual plot
ax = axes[1,0]
residuals = y_wtp_te - y_wtp_pred
ax.scatter(y_wtp_pred, residuals, alpha=0.4, s=15, color=GOLD, edgecolors='none')
ax.axhline(0, color='red', linestyle='--', lw=1.5)
style_ax(ax, 'Residual Plot — WTP\n(Gradient Boosting)', 'Predicted WTP', 'Residuals')

# Feature importance RF for WTP
ax = axes[1,1]
fi_top = pd.Series(reg_results['rf_fi_wtp']).sort_values()
colors_fi = [TEAL if 'Eng_' in k else GOLD if 'Derived_' in k else NAVY for k in fi_top.index]
ax.barh(range(len(fi_top)), fi_top.values, color=colors_fi, edgecolor='white')
ax.set_yticks(range(len(fi_top)))
ax.set_yticklabels([k.replace('Q13_','Pain: ').replace('Q16_','Feat: ').replace('Eng_','Eng: ')
                     .replace('Derived_','Der: ') for k in fi_top.index], fontsize=8)
style_ax(ax, 'Top Features — WTP Regression\n(Random Forest)', 'Importance', '')

# Distribution of WTP by income tier
ax = axes[1,2]
income_map = {1:'<5k',2:'5-10k',3:'10-20k',4:'20-40k',5:'>40k',6:'PNS'}
plot_df = master.copy()
plot_df['Income_Label'] = plot_df['Q5_IncomeTier'].map(income_map)
order = ['<5k','5-10k','10-20k','20-40k','>40k','PNS']
means = plot_df.groupby('Income_Label')['Q21_WTP_AED'].mean().reindex(order)
bars = ax.bar(range(len(means)), means.values, color=PALETTE[:len(means)], edgecolor='white')
ax.set_xticks(range(len(means)))
ax.set_xticklabels(means.index, fontsize=8, rotation=20, ha='right')
for bar, val in zip(bars, means.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'AED {val:.0f}',
            ha='center', va='bottom', fontsize=7, fontweight='bold')
style_ax(ax, 'Mean WTP by Income Tier', 'Income Tier (AED)', 'Mean WTP (AED/month)')

save_fig("11_regression_results")
print("   Fig 11 saved")

results['regression'] = reg_results

# ══════════════════════════════════════════════════════════════════════════════
# 8. BUSINESS FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] Business Findings Summary...")

# Most common pricing preference
pricing_map = {1:'Free with Ads',2:'Freemium',3:'Monthly Sub',4:'Annual Sub',5:'One-time',6:'Pay-per-feature'}
pricing_dist = master['Q22_PricingModel'].map(pricing_map).value_counts()

# Highest priority features
feature_cols_16 = [c for c in master.columns if c.startswith('Q16_')]
feat_means = master[feature_cols_16].mean().sort_values(ascending=False)
feat_name_map = {
    'Q16_AIExpenseCategorisation':'AI Expense Categorisation',
    'Q16_MultiCurrencyTracking':'Multi-Currency Tracking',
    'Q16_SavingsGoals':'Savings Goals',
    'Q16_RemittanceBestRate':'Remittance Best Rate',
    'Q16_InvestmentTracking':'Investment Tracking',
    'Q16_SubscriptionAlerts':'Subscription Alerts',
    'Q16_CreditScore':'Credit Score',
    'Q16_HealthDashboard':'Health Dashboard',
    'Q16_ChatbotQueries':'Chatbot Queries',
    'Q16_GamifiedSavings':'Gamified Savings',
}
feat_means.index = feat_means.index.map(feat_name_map)

# Fig — Business findings
fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor='white')
fig.suptitle('FinWise AI — Business Feasibility & Strategy Findings', fontsize=14, fontweight='bold', color=NAVY)

# Feature priority
ax = axes[0,0]
colors = [TEAL if i < 3 else GOLD if i < 6 else LGRAY for i in range(len(feat_means))]
ax.barh(range(len(feat_means))[::-1], feat_means.values, color=colors, edgecolor='white')
ax.set_yticks(range(len(feat_means)))
ax.set_yticklabels(feat_means.index[::-1], fontsize=8)
ax.axvline(3.5, color='red', linestyle='--', lw=1, alpha=0.7, label='High priority (>3.5)')
ax.legend(fontsize=8)
style_ax(ax, 'Feature Priority Ranking\n(Mean desirability score 1-5)', 'Mean Score', '')

# Pricing model preference
ax = axes[0,1]
pricing_order = pricing_dist.sort_values(ascending=False)
pie_colors = [TEAL,GOLD,NAVY,CORAL,SAGE,'#8B5CF6'][:len(pricing_order)]
wedges, texts, autotexts = ax.pie(
    pricing_order.values, labels=pricing_order.index,
    colors=pie_colors, autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(width=0.65), textprops={'fontsize':8},
    pctdistance=0.75
)
for at in autotexts:
    at.set_fontsize(7)
ax.set_title('Pricing Model Preference\n(Q22)', fontsize=12, fontweight='bold', color=NAVY)

# Segment WTP vs Adoption readiness bubble chart
ax = axes[1,0]
seg_data = master.groupby('Persona_Cluster').agg(
    wtp=('Q21_WTP_AED','mean'),
    readiness=('Eng_AdoptionReadiness','mean'),
    pain=('Derived_PainComposite','mean'),
    n=('Respondent_ID','count')
).reset_index()
for _, row in seg_data.iterrows():
    color = PERSONA_COLORS.get(row['Persona_Cluster'], NAVY)
    ax.scatter(row['readiness'], row['wtp'], s=row['n']*0.8,
               color=color, alpha=0.75, edgecolors='white', linewidth=1.5)
    ax.annotate(row['Persona_Cluster'].replace(' ','\n'),
                (row['readiness'], row['wtp']),
                fontsize=7, ha='center', va='bottom',
                xytext=(0, 8), textcoords='offset points')
style_ax(ax, 'Segment Opportunity Map\n(bubble size = n respondents)',
         'Adoption Readiness (0-1)', 'Mean WTP (AED/month)')
ax.axhline(seg_data['wtp'].mean(), color='gray', linestyle=':', lw=1)
ax.axvline(seg_data['readiness'].mean(), color='gray', linestyle=':', lw=1)
ax.text(seg_data['readiness'].mean()+0.002, seg_data['wtp'].max()*0.98,
        '← High readiness + High WTP\n   = Priority segment', fontsize=7, color=NAVY)

# TAM waterfall
ax = axes[1,1]
total_n = 1200
wtp_n     = master['Q20_WTP_Binary'].sum()
high_conf = (master['Q31_ConfideanceAI'] >= 7).sum() if 'Q31_ConfideanceAI' in master.columns else (master['Q31_ConfidenceAI'] >= 7).sum()
target    = (master['Flag_ValueSeeker'] == 1).sum()

stages = ['Total\nRespondents','Willing\nto Pay','High AI\nConfidence','Value\nSeekers']
values = [total_n, wtp_n, high_conf, target]
bar_c  = [LGRAY, TEAL, GOLD, CORAL]
bars = ax.bar(stages, values, color=bar_c, edgecolor='white', width=0.55)
for bar, val in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8, str(val),
            ha='center', va='bottom', fontsize=10, fontweight='bold', color=NAVY)
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
            f'{val/total_n*100:.0f}%',
            ha='center', va='center', fontsize=9, color='white', fontweight='bold')
style_ax(ax, 'Demand Funnel\n(from survey sample)', '', 'Count')

save_fig("12_business_findings")
print("   Fig 12 saved")

results['business'] = {
    'top_features':         feat_means.head(5).round(2).to_dict(),
    'pricing_preferred':    pricing_dist.idxmax(),
    'pricing_dist':         pricing_dist.to_dict(),
    'wtp_willing_pct':      round(master['Q20_WTP_Binary'].mean()*100, 1),
    'median_wtp':           round(master['Q21_WTP_AED'].median(), 1),
    'value_seeker_count':   int(master['Flag_ValueSeeker'].sum()),
    'high_conf_count':      int(high_conf),
}

# ══════════════════════════════════════════════════════════════════════════════
# SAVE ALL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with open(f"{OUT_DIR}analysis_results.json", 'w') as f:
    # Clean non-serialisable types
    def clean(obj):
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        if isinstance(obj, dict):            return {k: clean(v) for k,v in obj.items()}
        if isinstance(obj, list):            return [clean(v) for v in obj]
        return obj
    json.dump(clean(results), f, indent=2)

print("\n✅ All analysis complete. Outputs saved to outputs/")
print(f"   Figures: {len([f for f in os.listdir(FIG_DIR) if f.endswith('.png')])}")
print(f"   Models:  {len([f for f in os.listdir(OUT_DIR) if f.endswith('.pkl')])}")
print(f"   CSVs:    {len([f for f in os.listdir(OUT_DIR) if f.endswith('.csv')])}")
