package(
    default_visibility = ["//visibility:public"],
)

load("//:rules.bzl", "lt_nodejs_library")

py_binary(
    name = "run_web_server",
    srcs = ["run_web_server.py"],
    data = [
        "//sdk2/visuals/javascript_tests/templates:d3_graph",
    ],
    main = "run_web_server.py",
    deps = [
        ":proto_to_json",
        "//sdk2/graph:lgf_graph",
    ],
)

py_binary(
    name = "proto_to_json",
    srcs = ["proto_to_json.py"],
    main = "proto_to_json.py",
    deps = [
        "//sdk2/analysis:plot_layer_error",
        "//sdk2/common:py_test_util",
        "//sdk2/graph/transform_graph:utils",
        "//sdk2/graph/transform_graph/graph_transformers:convert_to_debug_mode",
        "//sdk2/graph/transform_graph/graph_transformers:expand_subgraph_nodes",
        "//sdk2/proto:calibration_proto_py",
        "//sdk2/proto:dtypes_proto_py",
        "//sdk2/proto:graph_types_proto_py",
        "//sdk2/proto:hw_spec_proto_py",
        "//sdk2/proto:lgf_proto_py",
        "//sdk2/proto:ops_proto_py",
        "//sdk2/proto:performance_data_proto_py",
        "//sdk2/proto:sim_params_proto_py",
        "//sdk2/proto:sw_config_proto_py",
    ],
)
