from time import sleep

from ml_dash.run import RUN


def main():
  RUN.prefix = "geyang/scratch/some-experiment"
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


if __name__ == "__main__":
  main()
