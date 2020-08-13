package(
    default_visibility = ["//visibility:public"],
)

load("//:rules.bzl", "lt_nodejs_library")

py_binary(
    name = "plot_graph",
    srcs = ["plot_graph.py"],
    data = [
        "//graph_visualization/templates:d3_graph",
    ],
    main = "plot_graph.py",
    deps = [
        ":tf_proto_to_json",
    ],
)

py_binary(
    name = "tf_proto_to_json",
    srcs = ["tf_proto_to_json.py"],
    main = "tf_proto_to_json.py",
    deps = [
    ],
)