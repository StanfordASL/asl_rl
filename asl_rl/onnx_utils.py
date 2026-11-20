import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from asl_rl.model import ActorCritic
from asl_rl.normalizer import NORMALIZATION_EPS, NormalizerState

OPSET = 13
BATCH_DIM = "N"  # symbolic batch dimension in the ONNX graph
_TO_ONNX_ACT = {"relu": "Relu", "tanh": "Tanh", "elu": "Elu"}


def export_to_onnx(
    model: ActorCritic,
    params: dict,
    path: str,
    normalizer: NormalizerState = None,
    input_name: str = "obs",
) -> onnx.ModelProto:

    n_actor_layers = len(model.actor_hidden_dims) + 1
    layers = [
        (params[f"Dense_{i}"]["kernel"], params[f"Dense_{i}"]["bias"])
        for i in range(n_actor_layers)
    ]
    act_op = _TO_ONNX_ACT[model.activation]

    nodes, initializers = [], []
    current = input_name
    if normalizer is not None:
        norm_nodes, norm_inits, current = _normalization_nodes(normalizer, input_name)
        nodes += norm_nodes
        initializers += norm_inits
    last = len(layers) - 1
    for i, (kernel, bias) in enumerate(layers):
        w_name, b_name = f"W{i}", f"B{i}"
        initializers.append(
            numpy_helper.from_array(np.asarray(kernel, np.float32), w_name)
        )
        initializers.append(
            numpy_helper.from_array(np.asarray(bias, np.float32), b_name)
        )

        gemm_out = f"gemm{i}"
        nodes.append(
            helper.make_node(
                "Gemm", [current, w_name, b_name], [gemm_out], name=f"dense_{i}"
            )
        )
        current = gemm_out

        if i != last:
            act_out = f"act{i}"
            nodes.append(
                helper.make_node(act_op, [current], [act_out], name=f"{act_op}_{i}")
            )
            current = act_out

    in_dim = int(np.shape(layers[0][0])[0])
    out_dim = int(np.shape(layers[-1][0])[1])
    graph = helper.make_graph(
        nodes,
        "mlp",
        inputs=[
            helper.make_tensor_value_info(
                input_name, TensorProto.FLOAT, [BATCH_DIM, in_dim]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                current, TensorProto.FLOAT, [BATCH_DIM, out_dim]
            )
        ],
        initializer=initializers,
    )
    opsets = [helper.make_opsetid("", OPSET)]
    model = helper.make_model(
        graph, opset_imports=opsets, ir_version=helper.find_min_ir_version_for(opsets)
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return model


def _normalization_nodes(
    normalizer: NormalizerState, input_name: str
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto], str]:
    mean = np.asarray(normalizer.mean, np.float32)
    inv_std = (
        1.0 / np.sqrt(np.asarray(normalizer.var, np.float64) + NORMALIZATION_EPS)
    ).astype(np.float32)
    assert mean.ndim == 1 and mean.shape == inv_std.shape
    output_name = "obs_normalized"
    initializers = [
        numpy_helper.from_array(mean, "obs_mean"),
        numpy_helper.from_array(inv_std, "obs_inv_std"),
    ]
    nodes = [
        helper.make_node(
            "Sub", [input_name, "obs_mean"], ["obs_centered"], name="normalize_sub"
        ),
        helper.make_node(
            "Mul", ["obs_centered", "obs_inv_std"], [output_name], name="normalize_mul"
        ),
    ]
    return nodes, initializers, output_name
