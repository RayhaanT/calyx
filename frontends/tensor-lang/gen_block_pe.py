import calyx.builder as cb
from calyx import py_ast
from calyx.utils import bits_needed
from gen_pe import PE_NAME, BITWIDTH
from systolic_arg_parser import SystolicConfiguration

# Name of the block pe component
BLOCK_PE_NAME = "block_pe"

def instantiate_pe(comp: cb.ComponentBuilder, row: int, col: int):
    return comp.cell(f"pe_{row}_{col}", py_ast.CompInst(PE_NAME, []))

def get_pe_invoke(r, c, config: SystolicConfiguration):
    """
    gets the PE invokes for the PE at (r,c). mul_ready signals whether 1 or 0
    should be passed into mul_ready
    """

    top_ports = [ (
        f"top_{i}",
        py_ast.ThisPort(py_ast.CompVar(f"top_in_{c}_{i}"))
        ) for i in range(config.top_nz_width)
    ]
    if config.dbb is not None:
        # DBB
        top_di_ports = [ (
            f"top_di_{i}",
            py_ast.ThisPort(py_ast.CompVar(f"top_di_in_{c}_{i}"))
            ) for i in range(config.top_nz_width)
        ]
    else:
        top_di_ports = []
    left_ports = [ (
        f"left_{i}",
        py_ast.ThisPort(py_ast.CompVar(f"left_in_{r}_{i}"))
        ) for i in range(config.width)
    ]

    # Combine top_ports and left_ports, then add mul_ready
    combined_ports = top_ports + top_di_ports + left_ports
    combined_ports.append((
        "mul_ready",
        py_ast.ThisPort(py_ast.CompVar(f"mul_ready")),
    ))

    return py_ast.StaticInvoke(
        id=py_ast.CompVar(f"pe_{r}_{c}"),
        in_connects=combined_ports,
        out_connects=[],
    )

def block_pe(prog: cb.Builder, config: SystolicConfiguration):
    """
    Builds a "block" PE that contains a grid of Tensor PEs.
    Data is ingested from neighbouring block PEs or from the top/left of the grid.
    Data is broadcast to all tensor PEs within the block simultaneously.
    """
    # latency = ceil(log(config.width, 2)) + 1
    latency = 1
    comp = prog.component(name=BLOCK_PE_NAME, latency=latency)

    # TODO: figure out what to do with this
    comp.input("mul_ready", 1)

    # Instantiate input/outputs for top and left sides
    # Also the data-forwarding regiisters
    for col in range(config.tensor_top_length):
        for j in range(config.top_nz_width):
            in_port = comp.input(f"top_in_{col}_{j}", BITWIDTH)
            out_port = comp.output(f"top_out_{col}_{j}", BITWIDTH)
            reg = comp.reg(f"down_fwd_{col}_{j}", BITWIDTH)
            with comp.continuous as g:
                # out_port = reg.out doesn't work here but this does
                g.asgn(out_port, reg.out)
                reg.in_ = in_port
                reg.write_en = 1
            if config.dbb is not None:
                di_in_port = comp.input(f"top_di_in_{col}_{j}", config.di_bits)
                di_out_port = comp.output(f"top_di_out_{col}_{j}", config.di_bits)
                di_reg = comp.reg(f"down_fwd_di_{col}_{j}", config.di_bits)
                with comp.continuous as g:
                    g.asgn(di_out_port, di_reg.out)
                    di_reg.in_ = di_in_port
                    di_reg.write_en = 1

    for row in range(config.tensor_left_length):
        for j in range(config.width):
            in_port = comp.input(f"left_in_{row}_{j}", BITWIDTH)
            out_port = comp.output(f"left_out_{row}_{j}", BITWIDTH)
            reg = comp.reg(f"right_fwd_{row}_{j}", BITWIDTH)
            with comp.continuous as g:
                g.asgn(out_port, reg.out)
                reg.in_ = in_port
                reg.write_en = 1

    # Instantiate PEs and output ports so each accumulator value is visible from outside
    for row in range(config.tensor_left_length):
        for col in range(config.tensor_top_length):
            pe = instantiate_pe(comp, row, col)
            out_port = comp.output(f"final_{row}_{col}", BITWIDTH)
            with comp.continuous as g:
                g.asgn(out_port, pe.out)


    # Run all PEs in parallel
    par = py_ast.StaticParComp([
        get_pe_invoke(row, col, config)
            for row in range(config.tensor_left_length)
            for col in range(config.tensor_top_length)
        ])
    comp.control += par
