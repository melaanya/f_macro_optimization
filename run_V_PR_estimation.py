import argparse
from pathlib import Path

from numba.typed import List
import numpy as np

from src.estimation import estimate_grid
from src.io import format_output, load


OUTPUT_FOLDER = Path("./data/outputs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Linear approximation algorithm.")
    parser.add_argument("--db_file", type=str, required=True)
    parser.add_argument("--num_categs", type=int, required=True)
    parser.add_argument("--n_p", type=int, required=True)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--output_folder", type=Path, default=OUTPUT_FOLDER)
    args = parser.parse_args()

    raw_data = load(args.db_file)
    db_name = Path(args.db_file).stem
    data = List()
    [data.append(x) for x in raw_data]

    res_p, res_r = estimate_grid(
        data, num_categs=args.num_categs, n_p=args.n_p, beta=args.beta
    )

    output = format_output(res_p, res_r, args.n_p, args.beta)
    np.savetxt(
        OUTPUT_FOLDER
        / (
            f"python_{db_name}_{args.num_categs}_"
            f"{args.n_p}x{args.n_p}_{args.beta}.txt"
        ),
        output,
        delimiter=" ",
    )
