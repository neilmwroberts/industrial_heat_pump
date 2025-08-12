import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_paybacks(hp_price, hp_price_tc, hp_margin, gas_test, hp_test):
    # --- original helpers ---
    def years(q):
        try:
            return float(q.to('yr').m)
        except Exception:
            return float(q)

    def classify_payback(numer, denom):
        numer = numer.to('USD')
        denom = denom.to('USD/yr')
        n, d = numer.m, denom.m
        if d > 0:
            if n <= 0:
                return ("instant", None, "Instant")
            val = (numer / denom)
            rounded_val = round(years(val) * 2) / 2
            return ("finite", rounded_val, f"{rounded_val:.1f} yr")
        else:
            if n < 0:
                val = (numer / denom)
                rounded_val = round(years(val) * 2) / 2
                return ("finite", rounded_val, f"{rounded_val:.1f} yr")
            return ("none", None, "No payback")

    # --- original cases ---
    cases = [
        ("New Gas",                   hp_price - gas_test.capital_cost,                       gas_test.year_one_operating_costs - hp_test.year_one_operating_costs),
        ("Existing Gas",              hp_price,                                               gas_test.year_one_operating_costs - hp_test.year_one_operating_costs),
        ("New Gas + Tax Credit",      hp_price_tc - gas_test.capital_cost,                    gas_test.year_one_operating_costs - hp_test.year_one_operating_costs),
        ("Existing Gas + Tax Credit", hp_price_tc,                                            gas_test.year_one_operating_costs - hp_test.year_one_operating_costs),
    ]

    labels, vals, annos, colors = [], [], [], []
    for name, numer, denom in cases:
        kind, val, text = classify_payback(numer, denom)
        labels.append(name)
        annos.append(text)
        if kind == "finite":
            vals.append(val)
            colors.append("#4C78A8")
        elif kind == "instant":
            vals.append(0.2)
            colors.append("#59A14F")
        else:
            vals.append(0.2)
            colors.append("#E15759")

    # --- original plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors)

    ax.set_title(f"Payback Periods vs 45x Tax Credit (Gross Margin: {hp_margin:.0fP})", fontsize=18)
    ax.set_ylabel("Years", fontsize=16)
    ax.set_xticks(x)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_xticklabels(labels, fontsize=14, rotation=30, ha='right')
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for rect, txt in zip(bars, annos):
        ax.annotate(txt,
                    xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14)

    legend_elems = []
    if "#4C78A8" in colors:
        legend_elems.append(Patch(facecolor="#4C78A8", label="Finite payback"))
    if "#59A14F" in colors:
        legend_elems.append(Patch(facecolor="#59A14F", label="Instant (HP cheaper capex & opex)"))
    if "#E15759" in colors:
        legend_elems.append(Patch(facecolor="#E15759", label="No payback (HP higher capex & opex)"))
    ax.legend(handles=legend_elems, fontsize=12)

    plt.tight_layout()
    plt.show()
    return fig, ax
