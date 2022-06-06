import argparse
from pathlib import Path

from numba.typed import List
import numpy as np

from src.estimation import estimate_DV
from src.io import format_output, load


OUTPUT_FOLDER = Path("./data/outputs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Domain analysis algorithm.")
    parser.add_argument("--db_file", type=str, required=True)
    parser.add_argument("--num_categs", type=int, required=True)
    parser.add_argument("--n_angles", type=int, required=True)
    parser.add_argument("--output_folder", type=Path, default=OUTPUT_FOLDER)
    args = parser.parse_args()

    raw_data = load(args.db_file)
    db_name = Path(args.db_file).stem
    data = List()
    [data.append(x) for x in raw_data]

    hull = estimate_DV(
        data, num_categs=args.num_categs, num_angles=args.n_angles
    )

    np.savetxt(
        OUTPUT_FOLDER
        / (f"python_{db_name}_{args.num_categs}_{args.n_angles}.txt"),
        hull,
        delimiter=" ",
    )
