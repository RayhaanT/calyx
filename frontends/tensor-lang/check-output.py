#!/usr/bin/env python3
import numpy as np
import argparse
import json


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
    parser.add_argument("-A", "--tensor-left-length", type=int)
    parser.add_argument("-B", "--tensor-top-length", type=int)
    parser.add_argument("-C", "--tensor-width", type=int)
    parser.add_argument("-N", "--top-length", type=int)
    parser.add_argument("-M", "--left-length", type=int)
    parser.add_argument("-d", "--depth", type=int)
    parser.add_argument("-p", "--post-op", type=str, default=None)
    parser.add_argument("-j", "--json-file", type=str)

    args = parser.parse_args()

    A = args.tensor_left_length
    B = args.tensor_top_length
    C = args.tensor_width
    N = args.top_length
    M = args.left_length
    depth = args.depth
    post_op = args.post_op
    json_file = args.json_file

    top_length = B * N
    left_length = A * M

    left = np.zeros((left_length, depth), "f")
    top = np.zeros((depth, top_length), "f")
    json_data = json.load(open(json_file))["memories"]

    for block_row in range(M):
        for tensor_row in range(A):
            row = block_row * A + tensor_row
            for mult in range(C):
                for pos in range(depth//C):
                    col = pos * C + mult
                    left[row][col] = json_data[f"l{block_row}_{tensor_row}_{mult}"][pos]

    for block_col in range(N):
        for tensor_col in range(B):
            col = block_col * B + tensor_col
            for mult in range(C):
                for pos in range(depth//C):
                    row = pos * C + mult
                    top[row][col] = json_data[f"t{block_col}_{tensor_col}_{mult}"][pos]

    matmul_result = np.matmul(left, top)
    if post_op == "leaky-relu":
        matmul_result = np.where(matmul_result > 0, matmul_result, matmul_result * 0.01)
    elif post_op == "relu":
        matmul_result = np.where(matmul_result > 0, matmul_result, 0)

    json_result = np.zeros((B*N, M*A))
    for row in range(M*A):
        for tensor_col in range(B):
            for i in range(N):
                json_result[row][i*B + tensor_col] = json_data[f"out_mem_{row}_{tensor_col}"][i]

    if np.isclose(matmul_result, json_result, atol=1e-3).all():
        print("Correct")
    else:
        print("Incorrect\n. Should have been:\n")
        print(matmul_result)
        print("\nBut got:\n")
        print(json_result)
