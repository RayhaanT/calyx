import argparse
import json

SUPPORTED_POST_OPS = ["leaky-relu", "relu", "relu-dynamic"]


class SystolicConfiguration:
    """
    A class that represents a "systolic configuration". Includes:
    top_length, top_depth, left_length, left_depth
    post_op
    post_op has a default of None, the other values have no default: their value
    must be provided.
    """

    def parse_arguments(self):
        """
        Parses arguments to give self the following fields:
        top_length, top_depth, left_length, left_depth, and post_op
        """

        # Arg parsing
        parser = argparse.ArgumentParser(
            description="Generate an output-stationary systolic array in Calyx."
        )
        # Top and left lengths determine output array dimensions (thus input arr dims)
        # Other matrix dimension just determines the length of data stream (depth params)?
        #   Must be matched I think
        # Need new params for A, B, C block dimensions

        # top length should be equal to BN
        #   N tensor PEs that are B wide
        # left length = AM
        # Say W = BN, H = AM, output is HxW
        # Left matrix is WxR, right matrix is RxW
        # Each cell is the sum of R multiplies
        # So C | R
        # C determines how data is streamed to each side
        #   Bigger C -> fewer cycles needed to send all the requisite data
        # A,B determine how data is forwarded to multiple PE units at once
        #   Not every PE needs to wait for the clock edge to get the next piece of data
        #   Data is distributed to multiple PEs in a block at once.
        parser.add_argument("file", nargs="?", type=str)
        parser.add_argument("-A", "--tensor-left-length", type=int)
        parser.add_argument("-B", "--tensor-top-length", type=int)
        parser.add_argument("-C", "--tensor-width", type=int)
        parser.add_argument("-N", "--top-length", type=int)
        parser.add_argument("-M", "--left-length", type=int)
        parser.add_argument("-d", "--depth", type=int)
        parser.add_argument(
            "-p",
            "--post-op",
            help="post operation to be performed on the matrix-multiply result",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--fixed-dim",
            help=(
                "systolic array that only processes fixed dimension matrices."
                " By default, generated array can process matrices "
                " with any contraction dimension"
            ),
            action="store_true",
        )

        args = parser.parse_args()

        fields = [args.top_length, args.tensor_top_length, args.left_length, args.tensor_left_length, args.depth]
        if all(map(lambda x: x is not None, fields)):
            self.top_length = args.tensor_top_length * args.top_length
            self.top_depth = args.depth
            self.left_length = args.tensor_left_length * args. left_length
            self.left_depth = args.depth
            self.depth = args.depth
            self.post_op = args.post_op
            self.static = args.fixed_dim
            self.width = args.tensor_width
        elif args.file is not None:
            with open(args.file, "r") as f:
                spec = json.load(f)
                self.top_length = spec["top_length"]
                self.top_depth = spec["top_depth"]
                self.left_length = spec["left_length"]
                self.left_depth = spec["left_depth"]
                # default to not perform leaky_relu
                self.post_op = spec.get("post_op", None)
                # default to non-static (i.e., dynamic contraction dimension)
                self.static = spec.get("static", False)
        else:
            parser.error(
                "Need to pass either `FILE` or all of `"
                "-tl TOP_LENGTH -td TOP_DEPTH -ll LEFT_LENGTH -ld LEFT_DEPTH`"
            )
        # assert self.top_depth == self.left_depth, (
        #     f"Cannot multiply matrices: "
        #     f"{self.top_length}x{self.top_depth} and \
        #         {self.left_depth}x{self.left_length}"
        # )

    def get_output_dimensions(self):
        """
        Returns the dimensions of the output systolic array (in the form
        of num_rows x num_cols)
        """
        return (self.left_length, self.top_length)

    def get_contraction_dimension(self):
        """
        Returns the contraction dimension
        """
        assert (
            self.left_depth == self.top_depth
        ), "left_depth and top_depth should be same"
        # Could have also returend self.top_depth
        return self.left_depth

    def get_iteration_count(self):
        """
        Returns the iteration count if self.static
        Otherwise throws an error
        """
        # Could have also returend self.top_depth
        if self.static:
            (num_out_rows, num_out_cols) = self.get_output_dimensions()
            return self.get_contraction_dimension() + num_out_rows + num_out_cols + 4
        raise Exception(
            "Cannot get iteration count for systolic array with dynamic \
            contraction dimension"
        )
