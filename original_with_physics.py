import modulus
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from stl import mesh as np_mesh
from sympy import Symbol, Eq, Abs, sqrt, Max, sin, cos, Heaviside
from modulus.geometry import Geometry
from modulus.geometry.geometry import csg_curve_naming
from sympy import Symbol
from modulus.geometry.curve import SympyCurve
from modulus.geometry.primitives_2d import Polygon
from modulus.geometry.parameterization import Parameterization, Parameter, Bounds, OrderedParameterization
from modulus.loss.loss import CausalLossNorm
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    PointwiseConstraint,
)
from modulus.domain.validator import PointwiseValidator
from modulus.domain.monitor import PointwiseMonitor
from modulus.domain.inferencer import PointwiseInferencer
from modulus.eq.pdes.navier_stokes import NavierStokes
from morpho_equation_juntao import Morpho
#from something_equation import NavierStokesHydroMorpho
from modulus.geometry.tessellation import Tessellation
from modulus.key import Key
from modulus.geometry.helper import _sympy_sdf_to_sdf
from modulus.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)
import pandas as pd

m8 = np_mesh.Mesh.from_file("stl/8m.stl")

num_elem = m8.v0.shape[0]
all_edges = set()
repeat_edges = set()
for i in range(num_elem):
    x1 = frozenset({tuple(m8.v0[i][0:2]), tuple(m8.v1[i][0:2])})
    x2 = frozenset({tuple(m8.v1[i][0:2]), tuple(m8.v2[i][0:2])})
    x3 = frozenset({tuple(m8.v2[i][0:2]), tuple(m8.v0[i][0:2])})
    if x1 in all_edges:
        repeat_edges.add(x1)
    else:
        all_edges.add(x1)
    if x2 in all_edges:
        repeat_edges.add(x2)
    else:
        all_edges.add(x2)
    if x3 in all_edges:
        repeat_edges.add(x3)
    else:
        all_edges.add(x3)


boundary_edges = all_edges.difference(repeat_edges)

sea = set()
land = set()

for x in boundary_edges:
    y = list(x)
    if y[0][1] > 2.0e6:
        land.add(x)
    else:
        slope = abs((y[0][1]-y[1][1])/(y[0][0]-y[1][0]))
        if slope < 0.1:
            sea.add(x)
        else:
            land.add(x)

sea_vortex = set()
for edge in sea:
    for point in edge:
        if point not in sea_vortex:
            sea_vortex.add(tuple(k/100000. for k in point))
sea_vortex_list = list(sea_vortex)
sea_vortex_list = sorted(sea_vortex_list, key=lambda x:x[0])

land_vortex = set()
for edge in land:
    for point in edge:
        if point not in land_vortex:
            land_vortex.add(tuple(k/100000. for k in point))
land_vortex_list = list(land_vortex)
land_vortex_list = sorted(land_vortex_list, key=lambda x:x[0], reverse=True)

#coarser mesh
vortex_list_coarse = sea_vortex_list[0:-1:20] + land_vortex_list[0:-1:20]
sea_vortex_list = sea_vortex_list[0:-1:20]+[sea_vortex_list[-1]]
sea_edges = set()
for vortex1, vortex2 in zip(sea_vortex_list[:-1], sea_vortex_list[1:]):
    sea_edges.add((vortex1, vortex2))
land_vortex_list = land_vortex_list[0:-1:20]+[land_vortex_list[-1]]
land_edges = set()
for vortex1, vortex2 in zip(land_vortex_list[:-1], land_vortex_list[1:]):
    land_edges.add((vortex1, vortex2))

class PieceWiseLinearCurve(Geometry):
    def __init__(self, edges, parameterization=Parameterization()):
        s = Symbol(csg_curve_naming(0))
        x = Symbol("x")
        y = Symbol("y")
  
        min_x = float('-inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('inf')

        curve_parameterization = Parameterization({s: (0, 1)})
        curves = []
        for edge in edges:
            edge_list = list(edge)
            
            if max(edge_list[0][0], edge_list[1][0]) > max_x:
                max_x = max(edge_list[0][0], edge_list[1][0])
            if min(edge_list[0][0], edge_list[1][0]) < min_x:
                min_x = min(edge_list[0][0], edge_list[1][0])
            if max(edge_list[0][1], edge_list[1][1]) > max_y:
                max_y = max(edge_list[0][1], edge_list[1][1])
            if min(edge_list[0][1], edge_list[1][1]) < min_y:
                min_y = min(edge_list[0][1], edge_list[1][1])
            
            
            if edge_list[0][0] >= edge_list[1][0]:
                dx = edge_list[1][0] - edge_list[0][0]
                dy = edge_list[1][1] - edge_list[0][1]
            else:
                dx = edge_list[0][0] - edge_list[1][0]
                dy = edge_list[0][1] - edge_list[1][1]
            area = (dx**2+dy**2)**0.5
            normal_x = dy / area
            normal_y = dx / area
            if edge_list[0][0] >= edge_list[1][0]:
                line = SympyCurve(
                    functions={
                        "x": dx * s + edge_list[0][0], 
                        "y": dy * s + edge_list[0][1],
                        "normal_x": dy / area,
                        "normal_y": dx / area,
                    },
                    parameterization = curve_parameterization,                    
                    area = area,
                )
            else:
                line = SympyCurve(
                    functions={
                        "x": dx * s + edge_list[1][0], 
                        "y": dy * s + edge_list[1][1],
                        "normal_x": dy / area,
                        "normal_y": dx / area,
                    },
                    parameterization = curve_parameterization,                    
                    area = area,
                )
            curves.append(line)
        
        bounds = Bounds(
            {
                Parameter("x"): (min_x, max_x),
                Parameter("y"): (min_y, max_y),
            },
            parameterization = parameterization,
        )
        
        def _sdf():
            def sdf(invar, params, compute_sdf_derivatives=False):
                outputs = {}
                sdf_field = np.zeros_like(invar["x"])
                outputs["sdf"] = sdf_field

                return outputs
            return sdf


        super().__init__(
            curves, 
            _sdf(),
            dims=2,
            bounds=bounds,
            parameterization=parameterization,
        )

class ElevationsPlotter(ValidatorPlotter):
    "Define custom validator plotting class"

    def __call__(self, invar, true_outvar, pred_outvar):

        # only plot x,y dimensions
        invar = {k: v for k, v in invar.items() if k in ["x", "y"]}
        fs = super().__call__(invar, true_outvar, pred_outvar)
        return fs

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    morphs_eqn = Morpho(nu=0.01, rho=1.0, dim=2, time=True)
    elev_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("n")],
        layer_size=512,
        cfg=cfg.arch.fully_connected,
    )
    velo_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("v")],
        layer_size=512,
        cfg=cfg.arch.fully_connected,
    )
    sed_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("c")],
        cfg=cfg.arch.fully_connected,
    )
    # make list of nodes to unroll graph on
    #ns = NavierStokesHydroMorpho(nu=0.01, rho=1, g = 9.8, h = 0.2, e_s = 0.01, dim=2, time=True)

    nodes = morphs_eqn.make_nodes(detach_names=[                
                "u",
                "v",
              ]
            ) + [elev_net.make_node(name="elev_network"), 
            velo_net.make_node(name="velo_network"),
            sed_net.make_node(name="sed_network")]

    x, y, t = Symbol("x"), Symbol("y"), Symbol("t")
    sea_interior = Polygon(vortex_list_coarse)
    
    domain = Domain()

    df = pd.read_csv("/examples/hydro/pinns/original_simplified.csv")
    
    batch=256
    scaling=100000.

    # define time range
    time_length = 24
    time_range = {t: (0, time_length)}
    tide_amp = 1  # amplitude of the tide
    tide_t = 12  # period of the tide
    # make initial condition
    ic = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=sea_interior,
        outvar={
            "c": 0.0,
        },
        batch_size=100,
        parameterization={t: 0.0},
    )
    domain.add_constraint(ic, name="ic")

    # # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=sea_interior,
        outvar={"concentration": 0},
        lambda_weighting={"concentration": 0.1},
        batch_size=500,
        shuffle=False,
        loss=CausalLossNorm(eps=1.),
        parameterization=OrderedParameterization({t: (0, time_length)}, key=t),
    )
    domain.add_constraint(interior, "interior")

    for i in range(25):
        x_coords = np.expand_dims(df[df["t"]==i]["x"].to_numpy(), axis=-1)
        y_coords = np.expand_dims(df[df["t"]==i]["y"].to_numpy(), axis=-1)
        t_values = np.expand_dims(df[df["t"]==i]["t"].to_numpy(), axis=-1)
        h_values = np.expand_dims(df[df["t"]==i]["elev"].to_numpy(), axis=-1)
        u_values = np.expand_dims(df[df["t"]==i]["vel_x"].to_numpy(), axis=-1)
        v_values = np.expand_dims(df[df["t"]==i]["vel_y"].to_numpy(), axis=-1)
        c_values = np.expand_dims(df[df["t"]==i]["s"].to_numpy(), axis=-1)
        x_min = x_coords.min()
        y_min = y_coords.min()
        x_coords = (x_coords-x_min)/scaling
        y_coords = (y_coords-y_min)/scaling

        timestep_invar = {"x": x_coords,
                          "y": y_coords,
                          "t": t_values
                          }
        timestep_outvar = {"n": h_values, 
        "u": u_values, 
        "v": v_values, 
        "c": c_values
        }

        lambda_weighting={}
        lambda_weighting["n"]=np.full_like(timestep_invar["x"], 1.0)
        lambda_weighting["u"]=np.full_like(timestep_invar["x"], 1.0)
        lambda_weighting["v"]=np.full_like(timestep_invar["x"], 1.0)
        lambda_weighting["c"]=np.full_like(timestep_invar["x"], 10.0/batch)

        time_step = PointwiseConstraint.from_numpy(
            nodes=nodes,
            invar=timestep_invar,
            outvar=timestep_outvar,
            batch_size=batch,
            lambda_weighting=lambda_weighting,
        )
        domain.add_constraint(time_step, f"BC{i:04d}")

        validator = PointwiseValidator(
            nodes=nodes,
            invar=timestep_invar,
            true_outvar=timestep_outvar,
            batch_size=512,
            plotter=ElevationsPlotter(),
        )
        domain.add_validator(validator,  f"VAL{i:04d}")


    
    slv = Solver(cfg, domain)
    
    slv.solve()

if __name__ == "__main__":
    run()

