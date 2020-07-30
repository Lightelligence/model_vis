package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_d3_graph",
    srcs = ["test_d3_graph.py"],
    deps = [
        "//sdk2/common:py_test_util",
        "//sdk2/graph/tests:tf_mnist_sample",
        "//sdk2/graph/transform_graph/graph_transformers/tests:test_collapse_supported_subgraphs",
        "//sdk2/proto:lgf_proto_py",
        "//sdk2/proto:memory_layout_proto_py",
        "//sdk2/proto:subgraph_binary_proto_py",
        "//sdk2/visuals/javascript_tests:proto_to_json",
    ],
)
