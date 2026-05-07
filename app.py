import streamlit as st
import anthropic
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

# ── BRAND COLOURS ───────────────────────────────────────
COLOURS = {
    'background': '#FFFFFF',
    'text': '#1a1a2e',
    'axis': '#AAAAAA',
    'grid': '#F0F0F0',
    'palette': ['#B8DDB8', '#F2AABB', '#B8CDE8', '#F5D98B',
                '#C8B8E0', '#F5B8C4', '#C4D4B0', '#F5D4A8']
}

# ── STYLING ──────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    .stTextInput input { 
        border: 2px solid #B8DDB8;
        border-radius: 8px;
        font-size: 16px;
    }
    h1, h2, h3 { color: #1a1a2e !important; }
    .dataset-info {
        background: #F8F8F8;
        border-left: 4px solid #B8DDB8;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 13px;
        color: #555555;
        margin-bottom: 20px;
    }
    .insight-box {
        background: #F8F8F8;
        border-left: 4px solid #B8CDE8;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(
        r'D:\data_analysis\online_retail_II.csv',
        encoding='latin-1'
    )
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Revenue'] = df['Quantity'] * df['Price']
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df.dropna(subset=['Description'])
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    df = df[df['Quantity'] <= 1000]
    df = df[df['Price'] <= 500]
    df = df[df['Revenue'] <= 2000]
    df['Week'] = df['InvoiceDate'].dt.to_period('W-SUN').astype(str)
    df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    return df

# ── GENERATE CHART CODE ──────────────────────────────────
def generate_chart(user_request, df):
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    prompt = f"""You are a Python data analyst. You have access to a pandas 
dataframe called 'df' with these columns:
- Invoice: invoice number
- StockCode: product code
- Description: product name
- Quantity: units sold
- InvoiceDate: datetime of purchase
- Price: unit price in GBP
- Customer ID: customer identifier
- Country: customer country
- Revenue: Quantity x Price
- Week: week period (W-SUN format)
- Month: month period

The dataframe has {len(df):,} rows covering {df['InvoiceDate'].min().strftime('%b %Y')} to {df['InvoiceDate'].max().strftime('%b %Y')}.
Top countries: {', '.join(df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(5).index.tolist())}

The user wants: "{user_request}"

Write ONLY matplotlib Python code to visualise this. Rules:
- Do NOT include import statements
- Do NOT load or redefine df
- Use fig, ax = plt.subplots(figsize=(12, 6))
- Set fig.patch.set_facecolor('#FFFFFF')
- Set ax.set_facecolor('#FFFFFF')
- Use these colours in order: {COLOURS['palette']}
- Use '#1a1a2e' for text and titles
- Use '#AAAAAA' for axis labels
- Remove top and right spines
- Add a clear title
- End with plt.tight_layout()
- Do NOT include plt.show()
- Return only the code, no explanation"""

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

def generate_insights(user_request, df_filtered):
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    # Build chart-specific data summary based on what was requested
    request_lower = user_request.lower()

    # Determine what data to summarise based on the request
    if 'week' in request_lower:
        data_summary = df_filtered.groupby('Week')['Revenue'].sum().sort_index()
        data_context = f"Weekly revenue data:\n{data_summary.to_string()}"
        time_unit = "week"
    elif 'month' in request_lower:
        data_summary = df_filtered.groupby('Month')['Revenue'].sum().sort_index()
        data_context = f"Monthly revenue data:\n{data_summary.to_string()}"
        time_unit = "month"
    elif 'country' in request_lower or 'countr' in request_lower:
        data_summary = df_filtered.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10)
        data_context = f"Revenue by country:\n{data_summary.to_string()}"
        time_unit = "country"
    elif 'product' in request_lower or 'description' in request_lower:
        data_summary = df_filtered.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)
        data_context = f"Revenue by product:\n{data_summary.to_string()}"
        time_unit = "product"
    elif 'customer' in request_lower:
        data_summary = df_filtered.groupby('Customer ID')['Revenue'].sum().sort_values(ascending=False).head(10)
        data_context = f"Revenue by customer:\n{data_summary.to_string()}"
        time_unit = "customer"
    elif 'growth' in request_lower:
        data_summary = df_filtered.groupby('Month')['Revenue'].sum().pct_change() * 100
        data_context = f"Month over month growth %:\n{data_summary.to_string()}"
        time_unit = "growth period"
    else:
        data_summary = df_filtered.groupby('Month')['Revenue'].sum().sort_index()
        data_context = f"Monthly revenue data:\n{data_summary.to_string()}"
        time_unit = "period"

    # Check if December 2011 is in the filtered data
    has_partial_dec = '2011-12' in df_filtered['Month'].values

    partial_dec_note = """
IMPORTANT: December 2011 data is INCOMPLETE — it only covers 1-9 December 2011. 
Any comments about December 2011 must explicitly note it is a partial month.
Do NOT compare December 2011 directly to other full months without flagging this.
""" if has_partial_dec else ""

    prompt = f"""You are a senior data analyst presenting insights to business stakeholders.

The user asked to see: "{user_request}"

Here is the ACTUAL DATA shown in the chart:
{data_context}

{partial_dec_note}

Write 3-5 concise, specific, actionable bullet point insights based ONLY on the data above.
- Comment specifically on the {time_unit} level data shown — not broader dataset stats
- Use actual numbers from the data above
- Highlight the highest and lowest values
- Note any clear trends, spikes or drops
- Be specific — mention actual {time_unit} names/dates where relevant
- End with one "⚠️ Watch out:" or "💡 Opportunity:" point
- Keep each bullet to 1-2 lines maximum
- Format as bullet points starting with an emoji"""

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# ── MAIN APP ─────────────────────────────────────────────
st.title("📊 AI Data Analyst")
st.markdown("##### Natural language analytics — type what you want to see")

# ── DATASET INFO ─────────────────────────────────────────
st.markdown("""
<div class="dataset-info">
📦 <strong>Dataset:</strong> UCI Online Retail II — 
<a href="https://www.kaggle.com/code/olgaluzhetska/online-retail-cohort-analysis-and-other-stories/input" target="_blank">
View on Kaggle</a> &nbsp;|&nbsp; 
UK-based online retailer &nbsp;|&nbsp; 
Dec 2009 — Dec 2011 &nbsp;|&nbsp; 
1,067,371 transactions across 43 countries
</div>
""", unsafe_allow_html=True)

# Load data
with st.spinner("Loading data..."):
    df = load_data()

# ── TIME PERIOD SELECTOR ──────────────────────────────────
st.markdown("**Time period:**")
period = st.radio(
    "Select period",
    ["All Time", "Last 12 Months", "Last 6 Months", "Last 3 Months", "Last Month"],
    horizontal=True,
    label_visibility="collapsed"
)

# Filter df based on period
max_date = df['InvoiceDate'].max()

if period == "Last 12 Months":
    df_filtered = df[df['InvoiceDate'] >= max_date - pd.DateOffset(months=12)]
    df_prev = df[(df['InvoiceDate'] >= max_date - pd.DateOffset(months=24)) &
                 (df['InvoiceDate'] < max_date - pd.DateOffset(months=12))]
    df_yoy = None
    period_label = "vs previous 12 months"
    yoy_label = None

elif period == "Last 6 Months":
    df_filtered = df[df['InvoiceDate'] >= max_date - pd.DateOffset(months=6)]
    df_prev = df[(df['InvoiceDate'] >= max_date - pd.DateOffset(months=12)) &
                 (df['InvoiceDate'] < max_date - pd.DateOffset(months=6))]
    df_yoy = df[(df['InvoiceDate'] >= max_date - pd.DateOffset(months=18)) &
                (df['InvoiceDate'] < max_date - pd.DateOffset(months=12))]
    period_label = "vs previous 6 months"
    yoy_label = "YoY (same 6 months last year)"

elif period == "Last 3 Months":
    df_filtered = df[df['InvoiceDate'] >= max_date - pd.DateOffset(months=3)]
    df_prev = df[(df['InvoiceDate'] >= max_date - pd.DateOffset(months=6)) &
                 (df['InvoiceDate'] < max_date - pd.DateOffset(months=3))]
    df_yoy = df[(df['InvoiceDate'] >= max_date - pd.DateOffset(months=15)) &
                (df['InvoiceDate'] < max_date - pd.DateOffset(months=12))]
    period_label = "vs previous 3 months"
    yoy_label = "YoY (same 3 months last year)"

elif period == "Last Month":
    df_filtered = df[df['InvoiceDate'] >= max_date - pd.DateOffset(months=1)]
    df_prev = df[(df['InvoiceDate'] >= max_date - pd.DateOffset(months=2)) &
                 (df['InvoiceDate'] < max_date - pd.DateOffset(months=1))]
    df_yoy = df[(df['InvoiceDate'] >= max_date - pd.DateOffset(months=13)) &
                (df['InvoiceDate'] < max_date - pd.DateOffset(months=12))]
    period_label = "vs previous month"
    yoy_label = "YoY (same month last year)"

else:  # All Time
    df_filtered = df
    df_prev = None
    df_yoy = None
    period_label = None
    yoy_label = None

# ── HELPER: CALCULATE GROWTH ─────────────────────────────
def calc_growth(current, previous):
    if previous is None or len(previous) == 0:
        return None
    prev_val = previous['Revenue'].sum()
    if prev_val == 0:
        return None
    return ((current - prev_val) / prev_val) * 100

def calc_orders_growth(current, previous):
    if previous is None or len(previous) == 0:
        return None
    prev_val = previous['Invoice'].nunique()
    if prev_val == 0:
        return None
    return ((current - prev_val) / prev_val) * 100

def format_growth(value, label):
    if value is None or label is None:
        return ""
    arrow = "▲" if value >= 0 else "▼"
    color = "green" if value >= 0 else "red"
    return f"<span style='color:{color};font-size:11px'>{arrow} {abs(value):.1f}% {label}</span>"

# ── KPI ROW ──────────────────────────────────────────────
date_min = df_filtered['InvoiceDate'].min().strftime('%d %b %Y')
date_max = df_filtered['InvoiceDate'].max().strftime('%d %b %Y')
st.caption(f"📅 Showing: {date_min} — {date_max}")

# Calculate current period values
curr_revenue = df_filtered['Revenue'].sum()
curr_orders = df_filtered['Invoice'].nunique()
curr_aov = curr_revenue / curr_orders if curr_orders > 0 else 0
curr_countries = df_filtered['Country'].nunique()

# Calculate growth rates
rev_period_growth = calc_growth(curr_revenue, df_prev)
rev_yoy_growth = calc_growth(curr_revenue, df_yoy)
orders_period_growth = calc_orders_growth(curr_orders, df_prev)
orders_yoy_growth = calc_orders_growth(curr_orders, df_yoy)

prev_aov = (df_prev['Revenue'].sum() / df_prev['Invoice'].nunique()) if df_prev is not None and len(df_prev) > 0 else None
yoy_aov = (df_yoy['Revenue'].sum() / df_yoy['Invoice'].nunique()) if df_yoy is not None and len(df_yoy) > 0 else None
aov_period_growth = ((curr_aov - prev_aov) / prev_aov * 100) if prev_aov else None
aov_yoy_growth = ((curr_aov - yoy_aov) / yoy_aov * 100) if yoy_aov else None

# ── DISPLAY KPIs ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Revenue", f"£{curr_revenue/1e6:.2f}M")
    if period != "All Time":
        st.markdown(format_growth(rev_period_growth, period_label), unsafe_allow_html=True)
        if yoy_label:
            st.markdown(format_growth(rev_yoy_growth, yoy_label), unsafe_allow_html=True)

with col2:
    st.metric("Total Orders", f"{curr_orders:,}")
    if period != "All Time":
        st.markdown(format_growth(orders_period_growth, period_label), unsafe_allow_html=True)
        if yoy_label:
            st.markdown(format_growth(orders_yoy_growth, yoy_label), unsafe_allow_html=True)

with col3:
    st.metric("Avg Order Value", f"£{curr_aov:.2f}")
    if period != "All Time":
        st.markdown(format_growth(aov_period_growth, period_label), unsafe_allow_html=True)
        if yoy_label:
            st.markdown(format_growth(aov_yoy_growth, yoy_label), unsafe_allow_html=True)

with col4:
    st.metric("Countries", f"{curr_countries}")
    st.caption("Active in selected period")

st.divider()

# ── QUERY INPUT ───────────────────────────────────────────
user_request = st.text_input(
    "What would you like to see?",
    placeholder="e.g. Show me monthly revenue by top 5 countries",
    key="query_input"
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    generate_btn = st.button("✨ Generate", type="primary")

# ── EXAMPLE QUERIES ───────────────────────────────────────
st.markdown("**Try these:**")
examples = [
    "Weekly revenue trend",
    "Top 10 products by revenue",
    "Revenue by country",
    "Monthly growth %",
    "Top 10 customers by spend"
]
cols = st.columns(len(examples))
for i, example in enumerate(examples):
    with cols[i]:
        if st.button(example, key=f"ex_{i}"):
            user_request = example
            generate_btn = True

# ── GENERATE & DISPLAY ────────────────────────────────────
if generate_btn and user_request:
    
    col_chart, col_insights = st.columns([3, 1])
    
    with col_chart:
        with st.spinner("✨ Generating chart..."):
            try:
                code = generate_chart(user_request, df_filtered)
                code = code.replace("```python", "").replace("```", "").strip()
                local_vars = {
                    'df': df_filtered, 'plt': plt, 'pd': pd,
                    'np': np, 'mticker': mticker
                }
                exec(code, local_vars)
                st.pyplot(plt.gcf())
                plt.close()

                with st.expander("🔍 View generated code"):
                    st.code(code, language='python')

            except Exception as e:
                st.error(f"Error generating chart: {str(e)}")
                st.info("Try rephrasing your request")

    with col_insights:
        with st.spinner("💡 Generating insights..."):
            try:
                insights = generate_insights(user_request, df_filtered)
                st.markdown("### 💡 Key Insights")
                st.markdown(f"""
                <div class="insight-box">
                {insights.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")

# ── QUERY HISTORY ─────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

if generate_btn and user_request:
    if user_request not in st.session_state.history:
        st.session_state.history.append(user_request)

if st.session_state.history:
    st.divider()
    st.markdown("**Recent queries:**")
    for q in reversed(st.session_state.history[-5:]):
        st.markdown(f"• {q}")
