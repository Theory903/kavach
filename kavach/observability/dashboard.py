"""Kavach Security Dashboard (Phase D)

A Streamlit dashboard that visualizes the security posture in real time.
Reads from Kavach's generated logs and metrics.

Usage:
    streamlit run kavach/observability/dashboard.py
"""

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Kavach Security Operations", layout="wide", page_icon="🛡️")

def load_logs(log_path: str = "data/logs/kavach.log"):
    """Load JSON formatted logs."""
    p = Path(log_path)
    if not p.exists():
        return pd.DataFrame()
        
    records = []
    with open(p) as f:
        for line in f:
            if 'kavach.decision' in line:
                try:
                    
                    # Split default logger format "2026-03-03 17:00:00,000 [INFO] {"timestamp"..."
                    parts = line.strip().split("[INFO] ")
                    if len(parts) > 1:
                        data = json.loads(parts[1])
                        records.append(data)
                except Exception as e:
                    logging.warning(f"Parse error: {e}")
                    
    return pd.DataFrame(records)

def main():
    st.title("🛡️ Kavach Security Control Plane")
    
    with st.spinner("Loading telemetry..."):
        df = load_logs()

    if df.empty:
        st.warning("No security telemetry found. Start making requests to the Gateway!")
        return

    # Metrics Row
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    total_calls = len(df)
    blocked = len(df[df['decision'] == 'block'])
    allowed = len(df[df['decision'] == 'allow'])
    avg_latency = df['latency_ms'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Requests", f"{total_calls:,}")
    with col2:
        st.metric("Blocked Attacks", f"{blocked:,}", delta=f"{(blocked/max(total_calls,1))*100:.1f}%")
    with col3:
        st.metric("Allowed Requests", f"{allowed:,}")
    with col4:
        st.metric("Avg Latency", f"{avg_latency:.1f} ms")
        
    st.markdown("---")
    
    # Visualizations
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Decision Breakdown")
        decision_counts = df['decision'].value_counts()
        st.bar_chart(decision_counts)
        
    with col_chart2:
        st.subheader("Risk Score Distribution")
        st.line_chart(df['risk_score'])
        
    # Table of recent blocks
    st.subheader("Recent Blocked Requests")
    blocks_only = df[df['decision'] == 'block'].sort_values('timestamp', ascending=False)
    if not blocks_only.empty:
        # Expand nested identity
        blocks_only['role'] = blocks_only['identity'].apply(lambda x: x.get('role', 'unknown'))
        blocks_only['user'] = blocks_only['identity'].apply(lambda x: x.get('user_id', 'unknown'))
        
        display_df = blocks_only[['timestamp', 'session_id', 'user', 'role', 'risk_score', 'reasons', 'prompt']]
        st.dataframe(display_df.head(20), use_container_width=True)

if __name__ == "__main__":
    main()
