import argparse
from pathlib import Path

from numba.typed import List
import numpy as np

from src.estimation import estimate_grid, format_output
from src.load import load


OUTPUT_FOLDER = Path("./data/outputs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Linear approximation algorithm.")
    parser.add_argument("--num_categs", type=int, required=True)
    parser.add_argument("--n_p", type=int, required=True)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--output_folder", type=Path, default=OUTPUT_FOLDER)
    args = parser.parse_args()

    raw_data = load("./data/ESPGameNN.txt")
    data = List()
    [data.append(x) for x in raw_data]

    res_p, res_r = estimate_grid(
        data, 
        num_categs=args.num_categs,
        n_p=args.n_p, 
        beta=args.beta
    )

    output = format_output(res_p, res_r, args.n_p, args.beta)
    postfix = f"beta_{args.beta}" if args.beta != 1 else ""
    np.savetxt(
        OUTPUT_FOLDER / (
            f"python_ESPGameNN_{args.num_categs}_"
            f"{args.n_p}x{args.n_p}_{postfix}.txt"
        ),
        output,
        delimiter=" ",
    )
