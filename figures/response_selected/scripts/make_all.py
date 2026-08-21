"""Driver: rebuild every figure in this folder from ../data/ alone.

No checkpoints, no training/ tree, no GPU.  numpy + matplotlib + scikit-learn
(the last only to refit the 2-D decision regions of Fig. 5e from its saved
points).

  python make_all.py

The figures ship bare -- no figure title, subtitle, panel titles, provenance
stamp or explanatory prose -- so a text layer can be added downstream.  The one
exception, kept because it is the version already in use, is
fig_R22_R24_retiming_annotated, which this driver builds by rerunning the same
code with RESP_BARE=0.  Fig. 5d/5e are drawn by their original run's own code
and carry their original titles.

To refresh ../data/ from the run tree instead, see scripts/provenance/.
"""
import importlib
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    t0 = time.time()

    plt.rcdefaults()
    print("\n=== 5d / 5e ===")
    importlib.import_module("plot_fig5de").main()

    print("\n=== response-letter figures (bare) ===")
    S = importlib.import_module("style")
    S.apply()
    importlib.import_module("plot_baseline_hippo").main()
    importlib.import_module("plot_exp1").fig_heatmaps()
    importlib.import_module("plot_exp3").fig_retiming()
    importlib.import_module("plot_exp5").fig_lap_heatmap()
    importlib.import_module("plot_exp6").main()
    importlib.import_module("plot_fig3").main()

    print("\n=== fig_R22_R24_retiming, annotated ===")
    os.environ["RESP_BARE"] = "0"
    importlib.reload(S)
    exp3 = importlib.reload(importlib.import_module("plot_exp3"))
    exp3.fig_retiming()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.dirname(here)
    for ext in ("png", "pdf"):
        os.replace(os.path.join(out, f"fig_R22_R24_retiming.{ext}"),
                   os.path.join(out, f"fig_R22_R24_retiming_annotated.{ext}"))
    print("  renamed -> fig_R22_R24_retiming_annotated.png (+ .pdf)")

    os.environ["RESP_BARE"] = "1"
    importlib.reload(S)
    importlib.reload(exp3).fig_retiming()

    print(f"\n--- done in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()

