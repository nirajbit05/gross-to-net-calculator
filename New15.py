import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Gross to Net Pay Calculator | Expats", layout="wide")

# -----------------------------
# Custom CSS for aesthetics
# -----------------------------
st.markdown("""
<style>
    /* Add rounded corners and a slight shadow to containers */
    .st-emotion-cache-1r4qj8v {
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.02);
    }
    /* Style for the main block containers */
    [data-testid="stVerticalBlock"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Helpers
# -----------------------------
def fmt0(x):
    try:
        return f"{x:,.0f}"
    except Exception:
        return str(x)

def _as_float_or_none(v):
    """Coerce to float; treat empty/None/'none'/NaN as None."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("", "none", "nan"):
        return None
    try:
        # Allow '1_000_000' style and plain ints/floats
        return float(str(v).replace("_", ""))
    except Exception:
        return None

def sanitize_brackets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a tax bracket table to two columns: Upper_Limit (float|None) and Rate (0..1).
    Ensures a terminal row with Upper_Limit=None.
    Treats NaN like None to avoid NaN propagation in math.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame([{"Upper_Limit": None, "Rate": 0.0}])

    df2 = df.copy()

    # Coerce Upper_Limit robustly, mapping NaN -> None explicitly
    df2["Upper_Limit"] = df2["Upper_Limit"].apply(_as_float_or_none)

    # Coerce Rate, mapping >1 to percentages
    df2["Rate"] = pd.to_numeric(df2["Rate"], errors="coerce").fillna(0.0)
    df2.loc[df2["Rate"] > 1.0, "Rate"] = df2.loc[df2["Rate"] > 1.0, "Rate"] / 100.0

    # Sort with None/NaN treated as infinity
    def _key(u):
        return np.inf if (u is None or (isinstance(u, float) and np.isnan(u))) else float(u)

    df2 = df2.sort_values(by="Upper_Limit", key=lambda s: s.map(_key), ignore_index=True)

    # Ensure a terminal None row
    has_terminal = any((u is None) for u in df2["Upper_Limit"].tolist())
    if not has_terminal:
        last_rate = float(df2["Rate"].iloc[-1]) if len(df2) > 0 else 0.0
        df2.loc[len(df2)] = {"Upper_Limit": None, "Rate": last_rate}

    return df2

def apply_progressive(df: pd.DataFrame, amount: float):
    """Returns (total_tax, breakdown_rows_df)."""
    if amount <= 0:
        return 0.0, pd.DataFrame(columns=["From", "To", "Rate", "Amount", "Tax"])
    slabs = sanitize_brackets(df)
    total = 0.0
    rows = []
    remaining = amount
    lower = 0.0
    for _, row in slabs.iterrows():
        upper = row["Upper_Limit"]
        rate = float(row["Rate"])
        # Treat None/NaN as 'infinite' upper bound
        is_terminal = (upper is None) or (isinstance(upper, float) and np.isnan(upper))
        if is_terminal:
            slab_amt = max(remaining, 0.0)
        else:
            width = max(float(upper) - lower, 0.0)
            slab_amt = min(width, remaining)
        if slab_amt > 0:
            slab_tax = slab_amt * rate
            rows.append({"From": lower, "To": (None if is_terminal else float(upper)), "Rate": rate, "Amount": slab_amt, "Tax": slab_tax})
            total += slab_tax
            remaining -= slab_amt
        if remaining <= 0:
            break
        if not is_terminal:
            lower = float(upper)
    return total, pd.DataFrame(rows)

# -----------------------------
# Sidebar - Settings
# -----------------------------
st.sidebar.title("⚙️ Settings")

country = st.sidebar.selectbox(
    "Country of Work",
    ["Korea", "Taiwan", "China", "Singapore", "Japan", "India", "United States"],
    index=2,
)

show_usd = st.sidebar.toggle("Show USD Equivalent", value=True)
enable_export = st.sidebar.toggle("Enable CSV Export", value=False)

# Only allow US overlay when work country is NOT the US
us_overlay_allowed = (country != "United States")
us_overlay = st.sidebar.toggle("US Citizen/ GC Holder (Apply US Overlay)", value=False, disabled=not us_overlay_allowed)

# FX inputs (local -> USD). For the US, keep FX=1 and disable input.
st.sidebar.subheader("💱 Currency Exchange")
fx_defaults = {
    "Korea": 1_390.0,
    "Taiwan": 30.5,
    "China": 7.14,
    "Singapore": 1.28,
    "Japan": 155.0,
    "India": 87.0,
    "United States": 1.0,
}
fx_labels = {
    "Korea": "FX (KRW per USD)",
    "Taiwan": "FX (NTD per USD)",
    "China": "FX (CNY per USD)",
    "Singapore": "FX (SGD per USD)",
    "Japan": "FX (JPY per USD)",
    "India": "FX (INR per USD)",
    "United States": "FX (USD per USD)",
}
fx_disabled = (country == "United States")
fx = st.sidebar.number_input(fx_labels[country], min_value=0.0001, value=float(fx_defaults[country]), step=0.01, format="%.2f", disabled=fx_disabled)

# -----------------------------
# US Overlay (Federal) defaults in sidebar
# -----------------------------
st.sidebar.subheader("🇺🇸 US Federal (Overlay)")
feie = st.sidebar.number_input("FEIE (USD)", min_value=0, value=126_500, step=500, help="Foreign Earned Income Exclusion (applied for simple estimate)", disabled=(country == "United States"))
std_ded = st.sidebar.number_input("Standard Deduction (USD)", min_value=0, value=14_600, step=100)

st.sidebar.caption("Enter US bracket rates as decimals (e.g., 0.22 for 22%)")

us_brackets_default = pd.DataFrame([
    {"Upper_Limit": 11_600, "Rate": 0.10},
    {"Upper_Limit": 47_150, "Rate": 0.12},
    {"Upper_Limit": 100_525, "Rate": 0.22},
    {"Upper_Limit": 191_950, "Rate": 0.24},
    {"Upper_Limit": 243_725, "Rate": 0.32},
    {"Upper_Limit": 609_350, "Rate": 0.35},
    {"Upper_Limit": None, "Rate": 0.37},
])
us_brackets = st.sidebar.data_editor(us_brackets_default, num_rows="dynamic", use_container_width=True, key="us_editor")

# -----------------------------
# Title & Disclaimer
# -----------------------------
st.title("💰 Gross to Net Pay Calculator | Expats")
st.caption("Estimates only — excludes social insurance and detailed country/state rules. Consult a qualified tax professional for precise calculations.")

# -----------------------------
# Inputs (main panel)
# -----------------------------
with st.container(border=True):
    st.subheader("📝 Compensation Input")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        base = st.number_input("Base Pay", min_value=0.0, value=0.0, step=1_000.0, format="%.0f")
    with c2:
        var_pct = st.number_input("Variable % of Base", min_value=0.0, max_value=100.0, value=10.0, step=0.5, format="%.1f")
    with c3:
        crsu = st.number_input("Y1 CRSU", min_value=0.0, value=0.0, step=1_000.0, format="%.0f")
    with c4:
        signon = st.number_input("SignOn Bonus (if any)", min_value=0.0, value=0.0, step=1_000.0, format="%.0f")
    with c5:
        y1_rsu = st.number_input("Y1 RSU Value", min_value=0.0, value=0.0, step=1_000.0, format="%.0f")

variable_amt = base * (var_pct / 100.0)
earned = base + variable_amt + crsu + signon
total_comp = earned + y1_rsu

with st.container(border=True):
    st.subheader("📊 Compensation Summary")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.metric("💵 Total Comp (excl. Y1 RSU)", fmt0(earned))
    with cc2:
        st.metric("💵 Total Comp (incl. Y1 RSU)", fmt0(total_comp))


# -----------------------------
# Local layer placeholders and USD variables
# -----------------------------
local_tax = 0.0
local_net = total_comp

earned_usd = 0.0
rsu_usd = 0.0
local_tax_usd = 0.0
local_net_usd = 0.0

def show_local(currency_label: str, tax_value: float, net_value: float, fx_to_usd: float):
    cols = [1, 1] if show_usd and fx_to_usd > 0 else [1, 0.01]  # Hack to hide the second column
    lc, rc = st.columns(cols)

    with lc:
        with st.container(border=True):
            st.subheader(f"Local Layer ({currency_label})")
            st.metric(f"Local Tax ({currency_label})", fmt0(tax_value))
            st.metric(f"Net After Local Tax ({currency_label})", fmt0(net_value))

    if show_usd and fx_to_usd > 0:
        with rc:
            with st.container(border=True):
                st.subheader("Local Layer (USD)")
                st.metric("Local Tax (USD)", fmt0(tax_value / fx_to_usd))
                st.metric("Net After Local Tax (USD)", fmt0(net_value / fx_to_usd))

# -----------------------------
# Main Content with Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📍 Local Tax Calculation", "🇺🇸 US Overlay"])

with tab1:
    # -----------------------------
    # Country-specific local tax computations
    # -----------------------------
    if country == "Korea":
        st.subheader("🇰🇷 Korea")
        local_tax = total_comp * 0.21
        local_net = total_comp - local_tax
        show_local("KRW", local_tax, local_net, fx)
        if fx > 0:
            earned_usd = earned / fx
            rsu_usd = y1_rsu / fx
            local_tax_usd = local_tax / fx
            local_net_usd = local_net / fx

    elif country == "Taiwan":
        st.subheader("🇹🇼 Taiwan")
        st.caption("50% of earnings (excl. RSUs) >3M NTD is Tax Exempt")
        tw_default = pd.DataFrame([
            {"Upper_Limit": 590_000, "Rate": 0.05},
            {"Upper_Limit": 1_330_000, "Rate": 0.12},
            {"Upper_Limit": 2_660_000, "Rate": 0.20},
            {"Upper_Limit": 4_980_000, "Rate": 0.30},
            {"Upper_Limit": None, "Rate": 0.40},
        ])
        tw_brackets = st.data_editor(tw_default, num_rows="dynamic", use_container_width=True, key="tw_editor")

        above = max(earned - 3_000_000, 0.0)
        exempt = 0.5 * above
        taxable_earned = max(earned - exempt, 0.0)
        taxable_total = taxable_earned + y1_rsu

        local_tax, _ = apply_progressive(tw_brackets, taxable_total)
        local_net = total_comp - local_tax
        show_local("NTD", local_tax, local_net, fx)

        if fx > 0:
            earned_usd = earned / fx
            rsu_usd = y1_rsu / fx
            local_tax_usd = local_tax / fx
            local_net_usd = local_net / fx

    elif country == "China":
        st.subheader("🇨🇳 China")
        st.caption("A basic deduction of 60,000 CNY/year is applied before brackets")
        cn_default = pd.DataFrame([
            {"Upper_Limit": 36_000, "Rate": 0.03},
            {"Upper_Limit": 144_000, "Rate": 0.10},
            {"Upper_Limit": 300_000, "Rate": 0.20},
            {"Upper_Limit": 420_000, "Rate": 0.25},
            {"Upper_Limit": 660_000, "Rate": 0.30},
            {"Upper_Limit": 960_000, "Rate": 0.35},
            {"Upper_Limit": None, "Rate": 0.45},
        ])
        cn_brackets = st.data_editor(cn_default, num_rows="dynamic", use_container_width=True, key="cn_editor")
        taxable_total = max(total_comp - 60_000.0, 0.0)
        local_tax, _ = apply_progressive(cn_brackets, taxable_total)
        local_net = total_comp - local_tax
        show_local("CNY", local_tax, local_net, fx)
        if fx > 0:
            earned_usd = earned / fx
            rsu_usd = y1_rsu / fx
            local_tax_usd = local_tax / fx
            local_net_usd = local_net / fx

    elif country == "Singapore":
        st.subheader("🇸🇬 Singapore")
        sg_default = pd.DataFrame([
            {"Upper_Limit": 20_000, "Rate": 0.00},
            {"Upper_Limit": 30_000, "Rate": 0.02},
            {"Upper_Limit": 40_000, "Rate": 0.035},
            {"Upper_Limit": 80_000, "Rate": 0.07},
            {"Upper_Limit": 120_000, "Rate": 0.115},
            {"Upper_Limit": 160_000, "Rate": 0.15},
            {"Upper_Limit": 200_000, "Rate": 0.18},
            {"Upper_Limit": 240_000, "Rate": 0.19},
            {"Upper_Limit": 280_000, "Rate": 0.195},
            {"Upper_Limit": 320_000, "Rate": 0.20},
            {"Upper_Limit": 500_000, "Rate": 0.22},
            {"Upper_Limit": 1_000_000, "Rate": 0.23},
            {"Upper_Limit": None, "Rate": 0.24},
        ])
        sg_brackets = st.data_editor(sg_default, num_rows="dynamic", use_container_width=True, key="sg_editor")
        local_tax, _ = apply_progressive(sg_brackets, total_comp)
        local_net = total_comp - local_tax
        show_local("SGD", local_tax, local_net, fx)
        if fx > 0:
            earned_usd = earned / fx
            rsu_usd = y1_rsu / fx
            local_tax_usd = local_tax / fx
            local_net_usd = local_net / fx

    elif country == "Japan":
        st.subheader("🇯🇵 Japan")
        jp_default = pd.DataFrame([
            {"Upper_Limit": 1_950_000, "Rate": 0.05},
            {"Upper_Limit": 3_300_000, "Rate": 0.10},
            {"Upper_Limit": 6_950_000, "Rate": 0.20},
            {"Upper_Limit": 9_000_000, "Rate": 0.23},
            {"Upper_Limit": 18_000_000, "Rate": 0.33},
            {"Upper_Limit": 40_000_000, "Rate": 0.40},
            {"Upper_Limit": None, "Rate": 0.45},
        ])
        jp_brackets = st.data_editor(jp_default, num_rows="dynamic", use_container_width=True, key="jp_editor")
        local_tax, _ = apply_progressive(jp_brackets, total_comp)
        local_net = total_comp - local_tax
        show_local("JPY", local_tax, local_net, fx)
        if fx > 0:
            earned_usd = earned / fx
            rsu_usd = y1_rsu / fx
            local_tax_usd = local_tax / fx
            local_net_usd = local_net / fx

    elif country == "India":
        st.subheader("🇮🇳 India")
        st.caption("Includes flat ₹75,000 standard deduction, surcharge, and cess")

        # --- Flat Standard Deduction ---
        in_std_ded = 75_000.0

        # --- Progressive brackets (new regime illustrative) ---
        in_default = pd.DataFrame([
            {"Upper_Limit": 400_000, "Rate": 0.00},
            {"Upper_Limit": 800_000, "Rate": 0.05},
            {"Upper_Limit": 1_200_000, "Rate": 0.10},
            {"Upper_Limit": 1_600_000, "Rate": 0.15},
            {"Upper_Limit": 2_000_000, "Rate": 0.20},
            {"Upper_Limit": 2_400_000, "Rate": 0.25},
            {"Upper_Limit": None, "Rate": 0.30},
        ])

        # --- Apply flat standard deduction ---
        taxable_total = max(total_comp - in_std_ded, 0.0)

        # --- Base tax using progressive brackets ---
        in_brackets = st.data_editor(in_default, num_rows="dynamic", use_container_width=True, key="in_editor")
        base_tax, _ = apply_progressive(in_brackets, taxable_total)

        # --- Surcharge logic ---
        ti = taxable_total
        if ti > 20_000_000:
            surcharge_rate = 0.25
        elif ti > 10_000_000:
            surcharge_rate = 0.15
        elif ti > 5_000_000:
            surcharge_rate = 0.10
        else:
            surcharge_rate = 0.00
        surcharge = base_tax * surcharge_rate
        cess = 0.04 * (base_tax + surcharge)
        local_tax = base_tax + surcharge + cess
        local_net = total_comp - local_tax

        show_local("INR", local_tax, local_net, fx)
        if fx > 0:
            earned_usd = earned / fx
            rsu_usd = y1_rsu / fx
            local_tax_usd = local_tax / fx
            local_net_usd = local_net / fx

    else:  # United States
        st.subheader("🇺🇸 United States")

        # --- Apply Standard Deduction before Federal tax ---
        # Reuse the sidebar "US Federal (Overlay)" inputs already defined: std_ded, us_brackets
        taxable_total = max(total_comp - std_ded, 0.0)
        us_local_tax, _ = apply_progressive(us_brackets, taxable_total)

        # --- State tax selection (same choices & defaults as overlay) ---
        st.sidebar.subheader("🇺🇸 US State (Local)")
        state_choice = st.sidebar.selectbox(
            "Choose state",
            ["CA (Bay Area / Riverside)", "DC", "WA (Seattle)", "Other"],
            help="Flat % state tax applied to total comp"
        )
        if state_choice == "CA (Bay Area / Riverside)":
            state_rate = st.sidebar.number_input("CA rate", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.2f")
        elif state_choice == "DC":
            state_rate = st.sidebar.number_input("DC rate", min_value=0.0, max_value=1.0, value=0.08, step=0.01, format="%.2f")
        elif state_choice == "WA (Seattle)":
            state_rate = st.sidebar.number_input("Seattle (WA) rate", min_value=0.0, max_value=1.0, value=0.00, step=0.01, format="%.2f")
        else:
            state_rate = st.sidebar.number_input("Other states rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f")

        # --- Combine Federal + State ---
        state_tax = total_comp * state_rate
        local_tax = us_local_tax + state_tax
        local_net = total_comp - local_tax

        # --- Display + keep USD variables consistent for export ---
        show_local("USD", local_tax, local_net, 1.0)
        earned_usd = earned
        rsu_usd = y1_rsu
        local_tax_usd = local_tax
        local_net_usd = local_net

with tab2:
    # -----------------------------
    # US Overlay (Federal + State) for Citizens/GC outside the US
    # -----------------------------
    st.subheader("Federal + State Tax Overlay")
    if not us_overlay_allowed:
        st.info("US overlay is disabled when the work country is United States.")
    elif not us_overlay:
        st.info("Turn ON the 'US Citizen/ GC Holder' toggle in the sidebar to apply US federal and state taxes for expats")
    elif fx <= 0:
        st.error("FX rate must be > 0 to compute US overlay.")
    else:
        st.sidebar.subheader("🇺🇸 US State (Overlay)")
        state_choice = st.sidebar.selectbox(
            "Choose state",
            ["CA (Bay Area / Riverside)", "DC", "WA (Seattle)", "Other"],
            help="Used only when US overlay is ON. Applied as a flat % for estimate."
        )
        if state_choice == "CA (Bay Area / Riverside)":
            state_rate = st.sidebar.number_input("CA rate", min_value=0.0, max_value=1.0, value=0.10, step=0.005, format="%.3f")
        elif state_choice == "DC":
            state_rate = st.sidebar.number_input("DC rate", min_value=0.0, max_value=1.0, value=0.08, step=0.005, format="%.3f")
        elif state_choice == "WA (Seattle)":
            state_rate = st.sidebar.number_input("Seattle (WA) rate", min_value=0.0, max_value=1.0, value=0.00, step=0.005, format="%.3f")
        else:
            state_rate = st.sidebar.number_input("Other states rate", min_value=0.0, max_value=1.0, value=0.05, step=0.005, format="%.3f")

        # Calculations
        us_taxable = max((earned_usd + rsu_usd) - feie - std_ded, 0.0)
        us_tax, us_rows = apply_progressive(us_brackets, us_taxable)
        ftc_used = min(local_tax_usd, us_tax)
        us_due = max(us_tax - ftc_used, 0.0)
        state_tax_base = earned_usd + rsu_usd
        state_tax_usd = state_rate * state_tax_base
        combined_tax = local_tax_usd + us_due + state_tax_usd
        combined_net = (earned_usd + rsu_usd) - combined_tax

        # Display Results
        with st.container(border=True):
            st.subheader("🇺🇸 US Tax Summary (USD)")
            lc, rc = st.columns(2)
            with lc:
                st.metric("US Federal Tentative Tax", fmt0(us_tax))
                st.metric("Foreign Tax Credit Used", fmt0(ftc_used))
                st.metric("US Federal Tax Due", fmt0(us_due), delta=f"- {fmt0(ftc_used)} vs Tentative", delta_color="inverse")
                st.metric(f"US State Tax @ {state_rate:.1%}", fmt0(state_tax_usd))
            with rc:
                st.metric("Combined Tax (Local + US)", fmt0(combined_tax))
                st.metric("Combined Net Income", fmt0(combined_net))

# -----------------------------
# Export (CSV)
# -----------------------------
st.divider()
if enable_export:
    export_data = {
        "Country": country,
        "FX_to_USD": fx,
        "Base": base,
        "Variable_%": var_pct,
        "CRSU": crsu,
        "SignOn": signon,
        "Y1_RSU": y1_rsu,
        "Total_Comp_Local": total_comp,
        "Local_Tax_Local": local_tax,
        "Local_Net_Local": local_net,
    }
    # Add overlay data if calculated
    if us_overlay and us_overlay_allowed:
        export_data.update({
            "Total_Comp_USD": earned_usd + rsu_usd,
            "Local_Tax_USD": local_tax_usd,
            "US_Federal_Due_USD": us_due,
            "US_State_Tax_USD": state_tax_usd,
            "Combined_Tax_USD": combined_tax,
            "Combined_Net_USD": combined_net,
        })

    export_df = pd.DataFrame([export_data])

    st.download_button(
        "⬇️ Download Results as CSV",
        export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{country}_gross_to_net.csv",
        mime="text/csv",
        use_container_width=True,
    )
