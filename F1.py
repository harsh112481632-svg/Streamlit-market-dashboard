import xml.etree.ElementTree as ET

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from google import genai

GEMINI_API_KEY = "AIzaSyBzPmumpD_nukrjvXM0Y5zoHs798qsEHpQ"


try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.sidebar.error(f"Error initializing Gemini Client: {e}")


st.set_page_config(layout="wide", page_title="Market Dashboard")


@st.cache_data(
    ttl=300
)  # Caches data for 300 sec otherwise it reloads on every interaction which leads to rate limiting.
def get_data(ticker, period):
    """Fetches data and handles the timeline logic."""
    interval_map = {
        "1d": "5m",
        "5d": "90m",
        "1mo": "1d",
        "3mo": "1d",
        "6mo": "1d",
        "1y": "1d",
        "5y": "1wk",
        "max": "1mo",
    }
    selected_interval = interval_map.get(period, "1d")

    data = yf.download(
        ticker,
        period=period,
        interval=selected_interval,
    )
    if isinstance(
        data.columns, pd.MultiIndex
    ):  # Necessary otherwise charts breaks due to multiple headers
        data.columns = data.columns.get_level_values(0)
    return data


@st.cache_data(ttl=300)
def get_peer_comparison(tickers):
    """Fetches key ratios for a list of tickers to compare side-by-side."""
    data = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            data.append(
                {
                    "Ticker": t,
                    "Market Cap": info.get("marketCap", "N/A"),
                    "Forward P/E": info.get("forwardPE", "N/A"),
                    "Price/Sales": info.get("priceToSalesTrailing12Months", "N/A"),
                    "Debt/Equity": info.get("debtToEquity", "N/A"),
                    "ROE": info.get("returnOnEquity", "N/A"),
                    "Profit Margins": info.get("profitMargins", "N/A"),
                }
            )
        except Exception:
            continue

    if not data:
        return pd.DataFrame()

    comp_df = pd.DataFrame(data).set_index("Ticker")
    return comp_df


@st.cache_data(ttl=300)
def get_stock_info(ticker):
    """Fetches fundamental data for the stock."""
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except Exception as e:
        return {}


# Used web scraping instead of yfinance
@st.cache_data(ttl=300)
def scrape_market_data(url, count=10):
    # Scrapes Yahoo Finance tables (Gainers/Losers/Active)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check for request errors

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find(
            "table"
        )  # This works better than finding by class name as it finds the first html on the page.
        if not table:
            return pd.DataFrame()

        rows = table.find_all("tr")
        data = []

        for row in rows[1:]:
            if (
                len(data) >= count
            ):  # No of elements in data > count of elemenents required
                break
            cells = row.find_all("td")
            ticker = cells[0].text.strip()
            price = cells[3].text.strip()
            change = cells[4].text.strip()
            data.append({"Ticker": ticker, "Price": price, "Change": change})

        return pd.DataFrame(data)

    except Exception as e:
        st.error(f"Error scraping data: {e}")
        return pd.DataFrame()


with st.sidebar:  # compiled into .sidebar to reduce repetition
    st.title("Chart Settings")
    selected_ticker = st.text_input("Ticker", value="SPY").upper()
    compare_ticker = st.text_input("Compare with", value="").upper()
    time_frame = st.selectbox(
        "Timeframe",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "max"],
        index=0,  # Default to "1d"
    )
    chart_type = st.radio("Style", ["Candle", "Line"])

    st.subheader("Technical Indicators")
    show_sma_50 = st.checkbox("SMA 50 (Cyan)")
    show_sma_200 = st.checkbox("SMA 200 (Magenta)")

    st.markdown("---")
    st.markdown("**AI Features**")
    generate_summary = st.button("Generate AI Market Brief", type="primary")

st.title(f"Price Analysis: {selected_ticker}")

df = get_data(selected_ticker, time_frame)
df_compare = pd.DataFrame()
if compare_ticker:
    df_compare = get_data(compare_ticker, time_frame)

# Initialize metric variables to ensure availability for AI later
curr_price = 0
high_price = 0
low_price = 0
vol_today = 0

if not df.empty:
    fig = go.Figure()

    # Add Traces based on selection
    if chart_type == "Candle":
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=selected_ticker,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name=selected_ticker,
                line=dict(color="#ff9900", width=2),
                fill="tozeroy",  # area under the line is filled
                fillcolor="rgba(255, 0, 0, 0.1)",  # transparency
            )
        )

    if not df_compare.empty:
        fig.add_trace(
            go.Scatter(
                x=df_compare.index,
                y=df_compare["Close"],
                mode="lines",
                name=f"{compare_ticker} (Compare)",
                line=dict(
                    color="#ffffff", width=2, dash="dot"
                ),  # Dotted white line for contrast
                opacity=0.8,
            )
        )
    if show_sma_50:
        sma_50 = df["Close"].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=sma_50,
                mode="lines",
                name="SMA 50",
                line=dict(color="cyan", width=1.5),
            )
        )

    if show_sma_200:
        sma_200 = df["Close"].rolling(window=200).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=sma_200,
                mode="lines",
                name="SMA 200",
                line=dict(color="magenta", width=1.5),
            )
        )

    # Chart Styling
    fig.update_layout(  # To beautify and make it look like a real chart
        template="plotly_dark",
        height=600,
        margin=dict(
            l=10, r=10, t=10, b=10
        ),  # To remove extra white spaces that were there by default
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
    )

    # Render Chart
    st.plotly_chart(fig, use_container_width=True)

    # Render Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    curr_price = float(df["Close"].iloc[-1])
    high_price = float(df["High"].max())
    low_price = float(df["Low"].min())
    vol_today = int(df["Volume"].iloc[-1]) if not df["Volume"].empty else 0

    c1.metric("Close", f"${curr_price:.2f}")
    c2.metric("High", f"${high_price:.2f}")
    c3.metric("Low", f"${low_price:.2f}")
    c4.metric("Vol", f"{vol_today:,}")

    st.markdown("### Fundamentals")
    stock_info = get_stock_info(selected_ticker)

    if stock_info:
        m1, m2, m3, m4 = st.columns(4)

        # Helper to format large numbers (Billions/Trillions)
        market_cap = stock_info.get("marketCap", 0)
        if market_cap >= 1e12:
            mc_str = f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            mc_str = f"${market_cap/1e9:.2f}B"
        else:
            mc_str = f"${market_cap/1e6:.2f}M"

        pe_ratio = stock_info.get("trailingPE")
        pe_str = f"{pe_ratio:.2f}" if pe_ratio else "N/A"

        eps = stock_info.get("trailingEps")
        eps_str = f"${eps:.2f}" if eps else "N/A"

        beta = stock_info.get("beta")
        beta_str = f"{beta:.2f}" if beta else "N/A"

        m1.metric("Market Cap", mc_str)
        m2.metric("P/E Ratio", pe_str)
        m3.metric("EPS (TTM)", eps_str)
        m4.metric("Beta", beta_str)

    if compare_ticker:
        st.divider()
        st.subheader("⚔️ Peer Comparison")

        # Create a list of tickers to compare
        comparison_list = [selected_ticker, compare_ticker]

        peer_df = get_peer_comparison(comparison_list)

        if not peer_df.empty:
            # Display the table
            st.dataframe(
                peer_df,
                use_container_width=True,
                column_config={
                    selected_ticker: st.column_config.NumberColumn(format="%.2f"),
                    compare_ticker: st.column_config.NumberColumn(format="%.2f"),
                },
            )

else:
    st.error(
        f"No data available for {selected_ticker}. Please check the ticker symbol."
    )


st.divider()

st.subheader("Market Trends")

# Create three tabs for a clean look
tab1, tab2, tab3 = st.tabs(["Top Gainers", "Top Losers", "Most Active"])

# Define these outside tabs so that AI can use this even if tabs don't work
df_gainers = scrape_market_data("https://finance.yahoo.com/gainers")
df_losers = scrape_market_data("https://finance.yahoo.com/losers")
df_active = scrape_market_data("https://finance.yahoo.com/most-active")

with tab1:
    st.caption("Top gaining stocks today")
    st.dataframe(df_gainers, use_container_width=True, hide_index=True)

with tab2:
    st.caption("Top losing stocks today")
    st.dataframe(df_losers, use_container_width=True, hide_index=True)

with tab3:
    st.caption("Stocks with highest volume")
    st.dataframe(df_active, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Daily news (CNBC)")


def get_daily_news():
    url = "https://www.cnbc.com/id/10000664/device/rss/rss.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            news_items = []
            seen_titles = []

            for item in root.findall(".//item"):
                title = item.find("title")
                link = item.find("link")

                if title is not None and link is not None and title.text:
                    clean_title = title.text.strip()

                    if clean_title not in seen_titles:
                        news_items.append(
                            {"title": clean_title, "link": link.text.strip()}
                        )
                        seen_titles.append(clean_title)

            if news_items:
                return news_items[:5]  # Returns top 5 stories
    except Exception:
        return []
    return []


news_data = get_daily_news()

if news_data:
    for item in news_data:
        st.markdown(f"• [{item['title']}]({item['link']})")
else:
    st.error("⚠ Connection Error: Unable to retrieve live market news.")


comp_summary = "No competitor selected."
if compare_ticker and not df_compare.empty:
    comp_last = float(df_compare["Close"].iloc[-1])
    # Calculate percentage change over the selected period
    comp_start = float(df_compare["Open"].iloc[0])
    comp_pct = ((comp_last - comp_start) / comp_start) * 100
    comp_summary = f"{compare_ticker} is trading at ${comp_last:.2f}. Performance over this period: {comp_pct:+.2f}%"

fund_summary = "N/A"
if stock_info:
    fund_summary = (
        f"Market Cap: {stock_info.get('marketCap', 'N/A')}, "
        f"P/E Ratio: {stock_info.get('trailingPE', 'N/A')}, "
        f"Beta: {stock_info.get('beta', 'N/A')}"
    )

if generate_summary:
    st.divider()
    st.subheader(f"🤖 AI Market Brief: {selected_ticker}")

    trend_summary = ""
    if not df_gainers.empty:
        trend_summary += (
            f"\nTop Gainers: {', '.join(df_gainers['Ticker'].head().tolist())}"
        )
    if not df_losers.empty:
        trend_summary += (
            f"\nTop Losers: {', '.join(df_losers['Ticker'].head().tolist())}"
        )

    news_summary = ""
    if news_data:
        news_summary = "\n".join([f"- {n['title']}" for n in news_data[:5]])

    prompt = f"""
        You are a senior financial analyst. Based on the real-time data fetched below, provide a concise and insightful "Market Brief" for the user.
        
        **1. Focus Ticker ({selected_ticker}) Performance:**
        - Current Price: ${curr_price:.2f}
        - Period High: ${high_price:.2f}
        - Period Low: ${low_price:.2f}
        - Volume: {vol_today:,}
        
        **2. Fundamental Data:**
        {fund_summary}
        
        **3. Competitor Comparison:**
        {comp_summary}
        
        **4. Market Context:**
        {trend_summary}
        
        **5. Key News Headlines:**
        {news_summary}
        
        **Instructions:**
        - Interpret the price movement of {selected_ticker} (is it near highs/lows?).
        - Compare its performance against the competitor ({compare_ticker}) if available.
        - Use the fundamentals (P/E, Beta) to assess valuation or risk.
        - Mention any significant correlation with the news if apparent.
        - Keep the tone professional but engaging.
        - Use bullet points for readability.
        """

    try:
        # Updated API Call for google.genai library
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        st.markdown(response.text)
    except Exception as e:
        st.error(f"AI Generation Error: {e}")
