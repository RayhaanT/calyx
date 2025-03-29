import calyx.builder as cb
from calyx import py_ast
from calyx.utils import bits_needed

# Global constant for the current bitwidth.
BITWIDTH = 32
INTWIDTH = 16
FRACWIDTH = 16
# Name of the pe component
PE_NAME = "mac_pe"

def add_tree_layer(terms, comp):
    result = []
    n = len(terms)
    with comp.static_group(f"do_add{n}", 1):
        i = 0
        while i < n//2:
            adder = comp.fp_sop(f"add{n}_{i}", "add", BITWIDTH, INTWIDTH, FRACWIDTH)
            adder.left = terms[2*i].out
            adder.right = terms[2*i+1].out
            result.append(adder)
            i += 1
        if n % 2 == 1:
            result.append(terms[-1])
    return result

def pe(prog: cb.Builder, width: int):
    """
    Builds a vectorized "multiply and accumulate" PE that multiplies its `top`
    and `left` input values, and accumulate the value.
    The output is displayed through the `out` port.
    This PE can accept new inputs every cycle: therefore it has a `mul_ready`
    parameter to signal whether the output of the multiplier should be accumulated
    yet.
    """
    # Latency is the latency of the adder tree plus 1 to add to the acc
    # latency = ceil(log(width, 2)) + 1
    latency = 1
    comp = prog.component(name=PE_NAME, latency=latency)

    # Generate a multiplication unit for each input pair
    muls = []
    tops = []
    lefts = []
    for i in range(width):
        tops.append(comp.input(f"top_{i}", BITWIDTH))
        lefts.append(comp.input(f"left_{i}", BITWIDTH))
        muls.append(comp.pipelined_fp_smult(f"mul_{i}", BITWIDTH, INTWIDTH, FRACWIDTH))

    comp.input("mul_ready", 1)
    comp.output("out", BITWIDTH)
    acc = comp.reg("acc", BITWIDTH)

    this = comp.this()

    # Control group for all multiplications
    with comp.static_group("do_mul", 1):
        for i in range(width):
            muls[i].left = tops[i]
            muls[i].right = lefts[i]

    control_par = []
    control_par.append(py_ast.Enable(f"do_add{len(muls)}"))
    control_par.append(py_ast.Enable("do_mul"))

    to_add = muls
    # Build an adder tree
    # Not sure if this needs to be done in multiple cycles given "combinational" nature of adds
    while len(to_add) > 1:
        to_add = add_tree_layer(to_add, comp)
        if len(to_add) > 1:
            control_par.append(py_ast.Enable(f"do_add{len(to_add)}"))

    # Add add tree result to acc
    with comp.static_group(f"final_add", 1):
        adder = comp.fp_sop("add_final", "add", BITWIDTH, INTWIDTH, FRACWIDTH)
        adder.left = to_add[0].out
        adder.right = acc.out
        acc.write_en = this.mul_ready
        acc.in_ = adder.out

    control_par.append(py_ast.Enable(f"final_add"))
    comp.control += py_ast.StaticParComp(control_par)

    with comp.continuous:
        this.out = acc.out

def dbb_pe(prog: cb.Builder, width: int, top_nz_width: int, di_bits: int):
    """
    Builds a vectorized "multiply and accumulate" PE that multiplies its `top`
    and `left` input values, and accumulate the value.
    The top inputs are in DBB format, which means that the number of non-zero elements is bounded.
    Thus, multiplexers are used to select the appropriate `left` input,
    and only `top_nz_width` multipliers are needed.
    The output is displayed through the `out` port.
    This PE can accept new inputs every cycle: therefore it has a `mul_ready`
    parameter to signal whether the output of the multiplier should be accumulated
    yet.
    """
    # Latency is the latency of the adder tree plus 1 to add to the acc
    # But the adder tree is combinational, so the total latency is just 1
    latency = 1
    comp = prog.component(name=PE_NAME, latency=latency)

    # Top inputs and DBB indices
    tops = [
        comp.input(f"top_{i}", BITWIDTH)
        for i in range(top_nz_width)
    ]
    top_dis = [
        comp.input(f"top_di_{i}", di_bits)
        for i in range(top_nz_width)
    ]
    # Left inputs
    lefts = [
        comp.input(f"left_{i}", BITWIDTH)
        for i in range(width)
    ]

    # Generate a multiplication unit for each input pair
    muls = [
        comp.pipelined_fp_smult(f"mul_{i}", BITWIDTH, INTWIDTH, FRACWIDTH)
        for i in range(top_nz_width)
    ]

    comp.input("mul_ready", 1)
    comp.output("out", BITWIDTH)
    acc = comp.reg("acc", BITWIDTH)

    this = comp.this()

    # Control group for all multiplications
    with comp.static_group("do_mul", 1):
        for i in range(top_nz_width):
            muls[i].left = tops[i]
            for sel in range(width):
                muls[i].right = (top_dis[i] == cb.const(di_bits, sel)) @ lefts[sel]

    control_par = []
    control_par.append(py_ast.Enable("do_mul"))
    control_par.append(py_ast.Enable(f"do_add{len(muls)}"))

    to_add = muls
    # Build an adder tree
    while len(to_add) > 1:
        to_add = add_tree_layer(to_add, comp)
        if len(to_add) > 1:
            control_par.append(py_ast.Enable(f"do_add{len(to_add)}"))

    # Add add tree result to acc
    with comp.static_group(f"final_add", 1):
        adder = comp.fp_sop("add_final", "add", BITWIDTH, INTWIDTH, FRACWIDTH)
        adder.left = to_add[0].out
        adder.right = acc.out
        acc.write_en = this.mul_ready
        acc.in_ = adder.out

    control_par.append(py_ast.Enable(f"final_add"))
    comp.control += py_ast.StaticParComp(control_par)

    with comp.continuous:
        this.out = acc.out
