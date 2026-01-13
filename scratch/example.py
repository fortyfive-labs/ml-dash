# from ml_dash import Experiment
#
# with Experiment(
#   prefix="geyang/scratch/test-run",
#   remote="https://api.dash.ml",
#   local_path=None,
# ).run as exp:
#   for i in range(10):
#     exp.metrics("train").log(step=i, loss=1.0 / (i + 1))


# Or using the auto-start singleton:
from time import sleep

from ml_dash.run import RUN

if __name__ == "__main__":
  # EXP.PREFIX = f"geyang/scratch/quick-test/{Path(__file__).stem}"

  for k, v in vars(RUN).items():
    if k.startswith("_"):
      continue
    print(f"{k:>30}: {v}")

  from ml_dash.auto_start import dxp

  print(dxp)

  with dxp.run:
    for i in range(10):
      dxp.metrics("train").log(step=i, loss=1.0 / (i + 1))

      print("logged step", i)
      sleep(0.1)
