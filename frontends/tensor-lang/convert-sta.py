import numpy as np
import argparse
import json
import collections

if __name__ == "__main__":
    """
    This is a script to help you know whether the Calyx's systolic array
    generator is giving you the correct answers.

    How to use this script: run Calyx's systolic array generator and get an
    output json. Then run this script on the output json, and this script
    will check the answers against numpy's matrix multiplication implementation.

    Command line arguments are (no json support yet):
    -tl -td -ll -ld are the same as the systolic array arguments.
    -j which is the path to the json you want to check
    """
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-tl", "--top-length", type=int)
    parser.add_argument("-td", "--top-depth", type=int)
    parser.add_argument("-ll", "--left-length", type=int)
    parser.add_argument("-ld", "--left-depth", type=int)
    parser.add_argument("-p", "--post-op", type=str, default=None)
    parser.add_argument("-j", "--json-file", type=str)

    parser.add_argument("-A", "--tensor-left-length", type=int)
    parser.add_argument("-B", "--tensor-top-length", type=int)
    parser.add_argument("-C", "--tensor-width", type=int)
    parser.add_argument("-N", "--ttop-length", type=int)
    parser.add_argument("-M", "--tleft-length", type=int)
    parser.add_argument("-d", "--depth", type=int)
    parser.add_argument("-o", "--output", type=str)

    args = parser.parse_args()

    tl = args.top_length
    td = args.top_depth
    ll = args.left_length
    ld = args.left_depth
    post_op = args.post_op
    json_file = args.json_file

    assert td == ld, f"Cannot multiply matrices: " f"{tl}x{td} and {ld}x{ll}"

    left = np.zeros((ll, ld), "f")
    top = np.zeros((td, tl), "f")
    json_data = json.load(open(json_file))

    for r in range(ll):
        for c in range(ld):
            left[r][c] = json_data[f"l{r}"]["data"][c]

    for r in range(td):
        for c in range(tl):
            top[r][c] = json_data[f"t{c}"]["data"][r]

    fmt = json_data["l0"]["format"]

    A = args.tensor_left_length
    B = args.tensor_top_length
    C = args.tensor_width
    N = args.ttop_length
    M = args.tleft_length
    assert args.depth % C == 0
    depth = args.depth // C
    assert depth == td

    out = collections.defaultdict(lambda: {"data": [None] * depth, "format": fmt})
    for block_row in range(M):
        for tensor_row in range(A):
            row = block_row * A + tensor_row
            for mult in range(C):
                for pos in range(depth):
                    col = pos * C + mult
                    out[f"l{block_row}_{tensor_row}_{mult}"]["data"][pos] = float(left[row][col])

    for block_col in range(N):
        for tensor_col in range(B):
            col = block_col * B + tensor_col
            for mult in range(C):
                for pos in range(depth):
                    row = pos * C + mult
                    out[f"t{block_col}_{tensor_col}_{mult}"]["data"][pos] = float(top[row][col])

    with open(args.output, "w") as f:
        json.dump(out, f)
