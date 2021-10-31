import argparse
from pathlib import Path

from numba.typed import List
import numpy as np

from src.estimation import estimate_DV
from src.io import format_output, load


OUTPUT_FOLDER = Path("./data/outputs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Linear approximation algorithm.")
    parser.add_argument("--num_categs", type=int, required=True)
    parser.add_argument("--n_angles", type=int, required=True)
    parser.add_argument("--output_folder", type=Path, default=OUTPUT_FOLDER)
    args = parser.parse_args()

    raw_data = load("./data/ESPGameNN.txt")
    data = List()
    [data.append(x) for x in raw_data]

    hull = estimate_DV(
        data, num_categs=args.num_categs, num_angles=args.n_angles
    )

    np.savetxt(
        OUTPUT_FOLDER
        / (f"python_ESPGameNN_{args.num_categs}_{args.n_angles}.txt"),
        hull,
        delimiter=" ",
    )
