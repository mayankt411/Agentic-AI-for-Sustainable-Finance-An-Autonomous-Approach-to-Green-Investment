import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import requests

# ---------- Config ----------
DATA_PATH = "company_full_metrics.csv"
RANKED_OUT = "outputs/company_ranked_scores.csv"
PLOTS_DIR = "outputs/plots"
os.makedirs("outputs", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

st.set_page_config(layout="wide", page_title=" FINESSE", initial_sidebar_state="expanded")
# ---------- Local LLM via Ollama ----------
def hf_llm_answer(prompt: str, model: str = "tinyllama", max_tokens: int = 250, temperature: float = 0.0) -> str:
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "stream": False
        }
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response") or data.get("text") or str(data)
    except Exception as e:
        return f"Local LLM error: {e}"

# ---------- Smart Filters ----------
def filter_by_governance(df, governance_threshold=0.75, financial_threshold=0.7):
    return df[(df["Governance_Score"] >= governance_threshold) & (df["Financial_Score"] >= financial_threshold)].sort_values("Governance_Score", ascending=False)

def filter_by_esg(df, esg_threshold=0.75):
    return df[df["ESG_Score"] >= esg_threshold].sort_values("ESG_Score", ascending=False)

def filter_by_financials(df, financial_threshold=0.75):
    return df[df["Financial_Score"] >= financial_threshold].sort_values("Financial_Score", ascending=False)

def filter_by_balance(df, threshold=0.7):
    return df[(df["ESG_Score"] >= threshold) & (df["Financial_Score"] >= threshold)].sort_values("FinalScore", ascending=False)

def filter_by_risk(df, max_de_ratio=0.5, min_governance=0.7, min_dividend=0.02):
    return df[(df["Debt_to_Equity_Ratio"] <= max_de_ratio) & (df["Governance_Score"] >= min_governance) & (df["Dividend_Yield"] >= min_dividend)].sort_values("FinalScore", ascending=False)

def filter_by_dividend(df, min_dividend=0.03):
    return df[df["Dividend_Yield"] >= min_dividend].sort_values("Dividend_Yield", ascending=False)

# ---------- Prompt Builder ----------
def build_strict_prompt(user_question: str, df_scores: pd.DataFrame, top_k: int = None) -> str:
    cols = [
        "FinalScore", "ESG_Score", "Financial_Score", "Governance_Score",
        "Renewable_Energy_pct", "Environmental Risk Score", "Social Risk Score",
        "Market_Cap", "Debt_to_Equity_Ratio", "Dividend_Yield", "Revenue_Growth", "Sustainability_Impact"
    ]
    available_cols = [c for c in cols if c in df_scores.columns]
    q_lower = user_question.lower()

    if "governance" in q_lower:
        rows = filter_by_governance(df_scores).head(top_k or 10)
    elif "sustainability" in q_lower or "esg" in q_lower or "environment" in q_lower:
        rows = filter_by_esg(df_scores).head(top_k or 10)
    elif "financial" in q_lower:
        rows = filter_by_financials(df_scores).head(top_k or 10)
    elif "balance" in q_lower:
        rows = filter_by_balance(df_scores).head(top_k or 10)
    elif "risk" in q_lower:
        rows = filter_by_risk(df_scores).head(top_k or 10)
    elif "dividend" in q_lower:
        rows = filter_by_dividend(df_scores).head(top_k or 10)
    elif top_k:
        rows = df_scores.sort_values("FinalScore", ascending=False).head(top_k)
    else:
        rows = df_scores.copy()

    lines = []
    for _, r in rows.iterrows():
        parts = []
        for c in available_cols:
            val = r.get(c, None)
            if pd.isna(val): continue
            try: parts.append(f"{c}={float(val):.3f}")
            except: parts.append(f"{c}={val}")
        lines.append(f"{r.get('Company')}: " + ", ".join(parts))
    context = "\n".join(lines) if lines else "No rows in dataset."

    system = (
        "You are a sustainability-focused financial analyst. Your task is to analyze the dataset provided below "
        "and answer the user's question using only the data. You must reason from the values, compare companies, "
        "and extract insights. Do not invent information or refer to external sources. If data is missing, say so clearly."
    )

    return f"{system}\n\nDATASET:\n{context}\n\nUSER QUESTION: {user_question}\n\nAnswer concisely and insightfully."

# ---------- Data Load ----------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Debt_to_Equity Ratio": "Debt_to_Equity_Ratio",
        "Environmental Risk Score": "Environmental Risk Score",
        "Social Risk Score": "Social Risk Score"
    })
    return df

def fetch_financials_from_yf(tickers):
    rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            info = tk.info
            rows.append({
                "Company": t,
                "Stock_Price_Fetched": info.get("currentPrice"),
                "Market_Cap_Fetched": info.get("marketCap"),
                "Earnings_Per_Share_Fetched": info.get("trailingEps"),
                "PE_Ratio_Fetched": info.get("trailingPE"),
                "Debt_to_Equity_Ratio_Fetched": info.get("debtToEquity"),
                "Dividend_Yield_Fetched": (info.get("dividendYield") or 0) * 100.0
            })
        except:
            rows.append({k: None for k in ["Company", "Stock_Price_Fetched", "Market_Cap_Fetched", "Earnings_Per_Share_Fetched", "PE_Ratio_Fetched", "Debt_to_Equity_Ratio_Fetched", "Dividend_Yield_Fetched"]})
    return pd.DataFrame(rows)

# ---------- Scoring ----------
def compute_scores(df, weight_fin=0.5, weight_esg=0.5):
    esg_metrics = ["ESG_SCORE","Environmental Risk Score","Renewable_Energy_pct","Governance_Score","Social Risk Score"]
    fin_metrics = ["Stock_Price","Market_Cap","Revenue_Growth","Earnings_Per_Share","PE_Ratio","Debt_to_Equity_Ratio","Dividend_Yield"]
    for c in esg_metrics + fin_metrics:
        if c not in df.columns: df[c] = np.nan
    if df["Revenue_Growth"].dtype == object:
        df["Revenue_Growth"] = df["Revenue_Growth"].astype(str).str.replace("%","").replace("","0").astype(float) / 100.0
    df["Dividend_Yield"] = pd.to_numeric(df["Dividend_Yield"], errors='coerce').fillna(0)
    df.loc[df["Dividend_Yield"] > 1, "Dividend_Yield"] = df.loc[df["Dividend_Yield"] > 1, "Dividend_Yield"] / 100.0
    df[esg_metrics] = df[esg_metrics].astype(float).fillna(df[esg_metrics].median())
    df[fin_metrics] = df[fin_metrics].astype(float).fillna(df[fin_metrics].median())
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[esg_metrics + fin_metrics] = scaler.fit_transform(df_scaled[esg_metrics + fin_metrics])
    for c in ["Environmental Risk Score","Social Risk Score"]:
        if c in df_scaled.columns:
            df_scaled[c] = 1.0 - df_scaled[c]
    df_scaled["ESG_Score"] = df_scaled[esg_metrics].mean(axis=1)
    df_scaled["Financial_Score"] = df_scaled[fin_metrics].mean(axis=1)
    df_scaled["FinalScore"] = weight_fin * df_scaled["Financial_Score"] + weight_esg * df_scaled["ESG_Score"]
    df_scaled["Sustainability_Impact"] = (
        0.4 * df_scaled["ESG_Score"] +
        0.4 * df_scaled["Renewable_Energy_pct"] +
        0.2 * df_scaled["Environmental Risk Score"]
    )
    df_scaled["LastUpdated"] = datetime.utcnow().isoformat()
    df_scaled["Rank"] = df_scaled["FinalScore"].rank(ascending=False, method="min").astype(int)
    return df_scaled

# ---------- Allocation ----------
def recommend_allocations(df_scores, top_n=5, method="equal"):
    ranked = df_scores.sort_values("FinalScore", ascending=False).reset_index(drop=True)
    top = ranked.head(top_n).copy()
    if method == "equal":
        top["Weight"] = 1.0 / top_n
    elif method == "score":
        score_sum = top["FinalScore"].sum()
        top["Weight"] = top["FinalScore"] / score_sum if score_sum else 1.0/top_n
    return top[["Company","FinalScore","Weight","ESG_Score","Financial_Score","Sustainability_Impact","Rank"]]

# ---------- Explanation ----------
def explain_company(df_scores, company):
    row = df_scores[df_scores["Company"] == company]
    if row.empty:
        return f"Company '{company}' not found in dataset."
    r = row.iloc[0]
    expl = []
    expl.append(f"{company}: FinalScore={r['FinalScore']:.3f}, Rank={r['Rank']}")
    expl.append(f"ESG={r['ESG_Score']:.3f}, Financial={r['Financial_Score']:.3f}, Sustainability_Impact={r['Sustainability_Impact']:.3f}")
    metric_cols = ["ESG_SCORE","Environmental Risk Score","Renewable_Energy_pct","Governance_Score",
                   "Stock_Price","Market_Cap","Revenue_Growth","Earnings_Per_Share","PE_Ratio",
                   "Debt_to_Equity_Ratio","Social Risk Score","Dividend_Yield"]
    contrib = r[metric_cols].dropna().sort_values(ascending=False).head(3)
    expl.append("Top normalized contributors: " + ", ".join([f"{c}={contrib[c]:.3f}" for c in contrib.index]))
    return "\n".join(expl)

# ---------- Agentic Automation Trigger ----------
def run_agentic_pipeline(df_raw, weight_scheme="50-50", allocation_method="equal", top_n=5, fetch_new=True):
    if fetch_new:
        tickers = df_raw["Company"].tolist()
        fetched = fetch_financials_from_yf(tickers)
        df_merge = pd.merge(df_raw, fetched, on="Company", how="left")
        def prefer(orig_col, fetched_col):
            if orig_col in df_merge.columns and df_merge[orig_col].notnull().any():
                return df_merge[orig_col]
            return df_merge.get(fetched_col, pd.Series([np.nan]*len(df_merge)))
        df = df_merge.copy()
        df["Stock_Price"] = prefer("Stock_Price","Stock_Price_Fetched")
        df["Market_Cap"] = prefer("Market_Cap","Market_Cap_Fetched")
        df["Earnings_Per_Share"] = prefer("Earnings_Per_Share","Earnings_Per_Share_Fetched")
        df["PE_Ratio"] = prefer("PE_Ratio","PE_Ratio_Fetched")
        df["Debt_to_Equity_Ratio"] = prefer("Debt_to_Equity_Ratio","Debt_to_Equity_Ratio_Fetched")
        df["Dividend_Yield"] = prefer("Dividend_Yield","Dividend_Yield_Fetched")
    else:
        df = df_raw.copy()

    w_fin, w_esg = (0.5, 0.5) if weight_scheme == "50-50" else (0.6, 0.4)
    df_scores = compute_scores(df, weight_fin=w_fin, weight_esg=w_esg)
    alloc = recommend_allocations(df_scores, top_n=top_n, method=allocation_method)
    return df_scores, alloc

# ---------- UI ----------
st.title("FINESSE: Financial Intelligence for Navigated ESG Evaluation and Sustainable Search Engine")

st.sidebar.header("Agent Controls")
with st.sidebar.form("controls"):
    weight_choice = st.selectbox("Weighting scheme", options=["50-50", "60-40"])
    top_n = st.number_input("Top N for allocation", min_value=1, max_value=10, value=5)
    allocation_method = st.selectbox("Allocation method", options=["equal", "score"])
    lens = st.selectbox("Analytical Lens", options=["Financial View", "Sustainability View"])
    run_fetch = st.form_submit_button("Run Agentic Pipeline")

try:
    df_raw = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"CSV not found at {DATA_PATH}. Place your cleaned file there and refresh.")
    st.stop()

yf_fetch = st.sidebar.checkbox("Fetch latest financials", value=False)
if run_fetch:
    df_scores, alloc = run_agentic_pipeline(df_raw, weight_scheme=weight_choice, allocation_method=allocation_method, top_n=top_n, fetch_new=yf_fetch)
    df_scores.to_csv(RANKED_OUT, index=False)
    st.success("Pipeline run complete. Rankings saved.")
else:
    df_scores = pd.read_csv(RANKED_OUT) if os.path.exists(RANKED_OUT) else compute_scores(df_raw)
    alloc = recommend_allocations(df_scores, top_n=top_n, method=allocation_method)

# ---------- Dashboard ----------
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Ranked Companies (Top 20)")
    st.dataframe(df_scores.sort_values("FinalScore", ascending=False).reset_index(drop=True).head(20), height=400)

    st.subheader("Top-N Allocation Recommendation")
    st.table(alloc)

    top10 = df_scores.sort_values("FinalScore", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8,5))
    if lens == "Financial View":
        sns.barplot(y="Company", x="Financial_Score", data=top10, palette="Blues", ax=ax)
        ax.set_title("Top 10 by Financial Score")
    else:
        sns.barplot(y="Company", x="Sustainability_Impact", data=top10, palette="Greens", ax=ax)
        ax.set_title("Top 10 by Sustainability Impact")
    st.pyplot(fig)

with col2:
    st.subheader("Finance vs ESG")
    fig2, ax2 = plt.subplots(figsize=(6,5))
    sns.scatterplot(data=df_scores, x="Financial_Score", y="ESG_Score", s=100, ax=ax2)
    for i, r in df_scores.iterrows():
        if r["Rank"] <= 5:
            ax2.text(r["Financial_Score"]+0.01, r["ESG_Score"]+0.01, r["Company"], fontsize=9)
    ax2.set_xlabel("Financial Score")
    ax2.set_ylabel("ESG Score")
    st.pyplot(fig2)

    st.subheader("Quick Stats")
    st.metric("Top FinalScore", f"{df_scores['FinalScore'].max():.3f}")
    st.metric("Top Sustainability Impact", f"{df_scores['Sustainability_Impact'].max():.3f}")
    st.metric("Top Company", df_scores.sort_values("FinalScore", ascending=False).iloc[0]["Company"])

    

# ---------- Footer ----------
st.markdown("---")
st.caption("Agentic pipeline: fetch → score → recommend → narrate. Powered by dataset-grounded LLM and sustainability intelligence.")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Save current ranked CSV"):
        df_scores.to_csv(RANKED_OUT, index=False)
        st.success(f"Saved ranked CSV to {RANKED_OUT}")
with col_b:
    if st.button("Download top-N allocation JSON"):
        alloc_json = alloc.to_json(orient="records")
        st.download_button("Download JSON", data=alloc_json, file_name="topn_alloc.json")

# ---------- Chatbot ----------
st.markdown("---")
st.subheader("Agent Chat (ask about companies, sustainability, or allocation)")
chat_input = st.text_input("Ask the agent")

if st.button("Ask"):
    q = chat_input.strip()
    if q == "":
        st.info("Type a question")
    else:
        q_low = q.lower()
        if q_low.startswith("explain "):
            comp = q.split(" ", 1)[1].strip().upper()
            if comp in df_scores["Company"].values:
                expl = explain_company(df_scores, comp)
                st.text(expl)
                prompt = build_strict_prompt(f"Explain in 3 short bullet points why {comp} ranks where it does.", df_scores=df_scores, top_k=None)
                llm = hf_llm_answer(prompt, max_tokens=150, temperature=0.0)
                if llm:
                    st.write("LLM Summary:")
                    st.write(llm)
            else:
                st.write(f"Company '{comp}' not found in dataset.")
        elif "show esg" in q_low or "esg for" in q_low:
            parts = q.split()
            for p in parts:
                p_up = p.upper().strip()
                if p_up in df_scores["Company"].values:
                    row = df_scores[df_scores["Company"] == p_up].iloc[0]
                    st.write(row[["ESG_SCORE", "Environmental Risk Score", "Renewable_Energy_pct", "Governance_Score", "Social Risk Score", "Sustainability_Impact"]])
                    break
            else:
                st.write("Please include a company ticker (e.g., 'show ESG for TSLA').")
        else:
            prompt = build_strict_prompt(q, df_scores=df_scores, top_k=None)
            llm = hf_llm_answer(prompt, max_tokens=250, temperature=0.0)
            if llm:
                st.write("LLM response:")
                st.write(llm)
