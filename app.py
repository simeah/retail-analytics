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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# No custom styling - using standard Streamlit

# ── LOAD DATA ────────────────────────────────────────────
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/simeah/retail-analytics/main/retail.parquet"
    df = pd.read_parquet(url)
    return df

# ── GENERATE CHART CODE ──────────────────────────────────
def generate_chart(user_request, df_filtered, period_name):
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

The dataframe has {len(df_filtered):,} rows covering {df_filtered['InvoiceDate'].min().strftime('%b %Y')} to {df_filtered['InvoiceDate'].max().strftime('%b %Y')}.
Time period: {period_name}
Top countries: {', '.join(df_filtered.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(5).index.tolist())}

The user wants: "{user_request}"

Write ONLY matplotlib Python code to visualise this. Rules:
- Do NOT include import statements
- Do NOT load or redefine df
- Use fig, ax = plt.subplots(figsize=(12, 6))
- Set fig.patch.set_facecolor('#FFFFFF')
- Set ax.set_facecolor('#FFFFFF')
- Use these colours in order: ['#B8DDB8', '#F2AABB', '#B8CDE8', '#F5D98B', '#C8B8E0', '#F5B8C4', '#C4D4B0', '#F5D4A8']
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

# ── GENERATE INSIGHTS ────────────────────────────────────
def generate_insights(user_request, df_filtered):
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    
    # Build analysis data that we KNOW works
    try:
        total_revenue = df_filtered['Revenue'].sum()
        total_orders = df_filtered['Invoice'].nunique()
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        unique_countries = df_filtered['Country'].nunique()
        unique_customers = df_filtered['Customer ID'].nunique()
        date_min = df_filtered['InvoiceDate'].min()
        date_max = df_filtered['InvoiceDate'].max()
        
        # Build relevant data based on the request
        request_lower = user_request.lower()
        
        analysis_data = f"""
DATASET SUMMARY:
- Total Revenue: £{total_revenue:,.2f}
- Total Orders: {total_orders:,}
- Average Order Value: £{avg_order_value:.2f}
- Unique Customers: {unique_customers:,}
- Active Countries: {unique_countries}
- Date Range: {date_min.strftime('%d %b %Y')} to {date_max.strftime('%d %b %Y')}
"""
        
        # Add specific analysis based on what was requested
        if 'revenue' in request_lower or 'monthly' in request_lower or 'weekly' in request_lower:
            monthly_data = df_filtered.groupby('Month')['Revenue'].sum().sort_index()
            analysis_data += f"\n\nMONTHLY REVENUE DATA:\n{monthly_data.to_string()}"
        
        if 'country' in request_lower:
            country_data = df_filtered.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10)
            analysis_data += f"\n\nTOP 10 COUNTRIES:\n{country_data.to_string()}"
        
        if 'product' in request_lower or 'description' in request_lower:
            product_data = df_filtered.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)
            analysis_data += f"\n\nTOP 10 PRODUCTS:\n{product_data.to_string()}"
        
        if 'customer' in request_lower:
            customer_data = df_filtered.groupby('Customer ID')['Revenue'].sum().sort_values(ascending=False).head(10)
            analysis_data += f"\n\nTOP 10 CUSTOMERS:\n{customer_data.to_string()}"
        
        if 'growth' in request_lower:
            monthly_growth = df_filtered.groupby('Month')['Revenue'].sum().pct_change() * 100
            analysis_data += f"\n\nMONTH-OVER-MONTH GROWTH %:\n{monthly_growth.to_string()}"
        
    except Exception as e:
        analysis_data = f"Basic metrics - Error in detailed analysis: {str(e)}"
    
    # Sonnet writes insights based on actual data
    insights_prompt = f"""You are a senior data analyst presenting insights to business stakeholders.

The user asked: "{user_request}"

Here is the actual data analysis:
{analysis_data}

Write 3-5 concise, specific bullet point insights based ONLY on the data above.
- Use actual numbers and values from the data
- Be specific and actionable
- Highlight trends, patterns, or anomalies
- Reference actual values
- End with one "⚠️ Watch out:" or "💡 Opportunity:" point
- Format as bullet points starting with emojis
- Keep each bullet to 1-2 lines maximum
- Do NOT mention technical errors or code issues
- Focus on the business insights from the actual numbers"""

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": insights_prompt}]
    )
    
    return message.content[0].text

# ── MAIN APP ─────────────────────────────────────────────

# HEADER
st.markdown("# 📊 AI Data Analyst")
st.markdown("##### Natural language analytics — type what you want to see")

# DATASET INFO
st.markdown("""
<div class="dataset-info">
📦 <strong>Dataset:</strong> UCI Online Retail II | UK-based online retailer | Dec 2009 — Dec 2011 | 1,067,371 transactions
</div>
""", unsafe_allow_html=True)

# ABOUT THIS DATA
with st.expander("📋 **About This Data**"):
    st.markdown("""
    **Order-level sales data for Dec 2009 - Dec 2011**
    *(Please note: partial month data for Dec 2011)*
    
    **Available fields:**
    - Invoice Number
    - Item Stock Code
    - Item Description
    - Quantity
    - Invoice Date
    - Price
    - Customer ID
    - Customer Country
    
    **How it works:**
    - 📊 **Charts** are generated instantly using AI
    - 💡 **Insights** analyse the chart data (10-15 sec to generate)
    - 🔍 Ask anything in natural language
    """)

# Load data
with st.spinner("Loading data..."):
    df = load_data()

# ── TIME PERIOD SELECTOR ─────────────────────────────────
max_date = df['InvoiceDate'].max()

st.markdown(f"<div style='font-size: 12px; color: #999999; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; margin-bottom: 8px;'>Time Period</div>", unsafe_allow_html=True)

period = st.selectbox(
    "Select Time Period",
    ["All Time", "Last 12 Months", "Last 6 Months", "Last 3 Months", "Last Month"],
    label_visibility="collapsed",
    key="period_select"
)

st.divider()

# ── FILTER DATA BASED ON PERIOD ──────────────────────────
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
    color = "#00AA66" if value >= 0 else "#DD5555"
    return f"<div style='color:{color}; font-size:10px; line-height: 1.4;'>{arrow} {abs(value):.1f}% {label}</div>"

# ── DISPLAY METRICS ──────────────────────────────────────
date_min = df_filtered['InvoiceDate'].min().strftime('%d %b %Y')
date_max = df_filtered['InvoiceDate'].max().strftime('%d %b %Y')

st.subheader(f"📈 Summary Metrics")
st.caption(f"Period: {date_min} — {date_max}")

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

# Display metrics in 4 columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    growth_text = f"{format_growth(rev_period_growth, period_label)}" if rev_period_growth is not None else "No previous data"
    st.metric("Total Revenue", f"£{curr_revenue/1e6:.2f}M", delta=growth_text if period_label else None)

with col2:
    growth_text = f"{format_growth(orders_period_growth, period_label)}" if orders_period_growth is not None else "No previous data"
    st.metric("Total Orders", f"{curr_orders:,}", delta=growth_text if period_label else None)

with col3:
    growth_text = f"{format_growth(aov_period_growth, period_label)}" if aov_period_growth is not None else "No previous data"
    st.metric("Avg Order Value", f"£{curr_aov:.2f}", delta=growth_text if period_label else None)

with col4:
    st.metric("Active Countries", curr_countries)

st.divider()

# ── QUERY SECTION ────────────────────────────────────────
st.subheader("🔍 What would you like to see?")
st.caption("Uses time period selected above")

user_request = st.text_input(
    "Your query",
    placeholder="e.g. Show me monthly revenue by top 5 countries",
    label_visibility="collapsed"
)

generate_btn = st.button("✨ Generate", use_container_width=True)

st.markdown("**Try these examples:**")
col1, col2, col3 = st.columns(3)

example_queries = [
    "Show me weekly revenue trend",
    "Top 10 products by revenue",
    "Revenue by country",
    "Monthly growth percentage",
    "Top 10 customers by spend",
    "Sales by country pie chart"
]

for i, example in enumerate(example_queries):
    col = col1 if i % 3 == 0 else (col2 if i % 3 == 1 else col3)
    with col:
        if st.button(example, key=f"ex_{i}", use_container_width=True):
            user_request = example
            generate_btn = True

st.divider()

# ── GENERATE & DISPLAY ────────────────────────────────────
if generate_btn and user_request:
    col_chart, col_insights = st.columns([2, 1])
    
    with col_chart:
        with st.spinner("✨ Generating chart..."):
            try:
                code = generate_chart(user_request, df_filtered, period)
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
                st.subheader("💡 Insights")
                st.info(insights)
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
