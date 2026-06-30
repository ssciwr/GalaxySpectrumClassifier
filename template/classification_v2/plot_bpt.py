"""
Plot the [NII] BPT diagram for the Cloudy HII (POPSTAR) and AGN model grids,
each in its own subplot.

The grid fluxes are normalised to Hbeta = 1 and the files contain no Halpha or
Hbeta column, so:
    y = log10([OIII]5007 / Hbeta) = log10(OIII_5007)            (Hbeta-normalised)
    x = log10([NII]6584  / Halpha) = log10(NII_6584 / 2.86)     (Case-B Ha/Hb)

Run headless-safe:
    MPLBACKEND=Agg python3 plot_bpt.py
"""

import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classificator_test import read_cloudy_grid

# Case-B recombination Halpha/Hbeta at T=1e4 K, n_e=100 cm^-3
HA_HB_CASE_B = 2.86


def bpt_ratios(df):
    """Return (log [NII]/Ha, log [OIII]/Hb) for a Hbeta-normalised grid table."""
    oiii = pd.to_numeric(df["OIII_5007"], errors="coerce")
    nii = pd.to_numeric(df["NII_6584"], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        oiii = oiii.where(oiii > 0, np.nan)
        nii = nii.where(nii > 0, np.nan)
        log_oiii_hb = np.log10(oiii)  # Hbeta = 1
        log_nii_ha = np.log10(nii / HA_HB_CASE_B)  # Halpha = 2.86 * Hbeta
    return log_nii_ha, log_oiii_hb


def draw_demarcations(ax):
    """Overplot the Kauffmann 2003 and Kewley 2001 [NII] BPT lines."""
    x = np.linspace(-2.0, 0.0, 500)
    ax.plot(x, 0.61 / (x - 0.05) + 1.3, "k--", lw=1.5, label="Kauffmann 2003")
    x2 = np.linspace(-2.0, 0.4, 500)
    ax.plot(x2, 0.61 / (x2 - 0.47) + 1.19, "k-", lw=1.5, label="Kewley 2001")


def main():
    hii_paths = sorted(glob.glob("C17_POPSTAR*.dat"))
    agn_paths = sorted(glob.glob("C17_AGN*.dat"))
    if not hii_paths or not agn_paths:
        raise FileNotFoundError("Expected C17_POPSTAR*.dat and C17_AGN*.dat grids.")

    df_hii = pd.concat([read_cloudy_grid(p) for p in hii_paths], ignore_index=True)
    df_agn = pd.concat([read_cloudy_grid(p) for p in agn_paths], ignore_index=True)
    print(f"HII grid points: {len(df_hii)}   AGN grid points: {len(df_agn)}")

    nii_ha_hii, oiii_hb_hii = bpt_ratios(df_hii)
    nii_ha_agn, oiii_hb_agn = bpt_ratios(df_agn)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

    axes[0].scatter(
        nii_ha_hii, oiii_hb_hii, s=12, c="tab:blue", alpha=0.4, edgecolor="none"
    )
    axes[0].set_title(f"HII / SF (POPSTAR, n={len(df_hii)})")

    axes[1].scatter(
        nii_ha_agn, oiii_hb_agn, s=12, c="tab:red", alpha=0.4, edgecolor="none"
    )
    axes[1].set_title(f"AGN (n={len(df_agn)})")

    for ax in axes:
        draw_demarcations(ax)
        ax.set_xlabel(r"log([NII]$\lambda6584$/H$\alpha$)")
        ax.set_xlim(-2.0, 0.6)
        ax.set_ylim(-1.5, 1.5)
        ax.legend(frameon=False, loc="lower left", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"log([OIII]$\lambda5007$/H$\beta$)")

    fig.suptitle("[NII] BPT diagram - Cloudy 17.00 model grids")
    fig.tight_layout()
    out = "bpt_nii.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
