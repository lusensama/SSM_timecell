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

    print("\n=== lapcount_performance ===")
    importlib.import_module("plot_lapcount").main()

    S = importlib.import_module("style")
    print("\n=== response-letter figures "
          + ("(textless) ===" if S.NOTEXT else "(bare) ==="))
    S.apply()
    importlib.import_module("plot_baseline_hippo").main()
    importlib.import_module("plot_exp1").fig_heatmaps()
    importlib.import_module("plot_exp3").fig_retiming()
    importlib.import_module("plot_exp5").fig_lap_heatmap()
    importlib.import_module("plot_exp6").main()
    importlib.import_module("plot_fig3").main()

    if S.NOTEXT:
        print("\n--- fig_R22_R24_retiming_annotated skipped (RESP_NOTEXT=1); "
              "build it with RESP_NOTEXT=0")
        print(f"\n--- done in {time.time() - t0:.1f}s")
        return

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

