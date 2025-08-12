# plot_lcoh.py
import numpy as np
import matplotlib.pyplot as plt

def _as_usd_per_mmbtu(x):
    try:
        return float(x.to('USD/MMBtu').m)
    except Exception:
        return float(x)

def plot_lcoh_breakdown(hp_test, gas_test, gas_location):
    labels = ["Heat Pump + Storage", "Heat Pump (No Storage)", "Gas"]

    capex = [
        _as_usd_per_mmbtu(hp_test.Levelized_Capex),
        _as_usd_per_mmbtu(hp_test.Levelized_Capex_no_storage),
        _as_usd_per_mmbtu(gas_test.Levelized_Capex),
    ]
    opex = [
        _as_usd_per_mmbtu(hp_test.Levelized_OpEx),
        _as_usd_per_mmbtu(hp_test.Levelized_OpEx_no_storage),
        _as_usd_per_mmbtu(gas_test.Levelized_OpEx),
    ]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.barh(y, capex, label="Levelized Capex")
    ax.barh(y, opex, left=capex, label="Levelized OpEx")

    # annotate totals
    totals = [c + o for c, o in zip(capex, opex)]
    for yi, total in zip(y, totals):
        ax.text(total * 1.005, yi, f"{total:.1f}", va='center', fontsize=12)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("USD per MMBtu", fontsize=12)
    ax.set_title(f"LCOH Breakdown – {gas_location}", fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(0, max(totals) * 1.15)

    plt.tight_layout()
    plt.show()
    return fig, ax
