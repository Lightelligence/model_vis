package(
    default_visibility = ["//visibility:public"],
)

load("//:rules.bzl", "lt_nodejs_library")

py_binary(
    name = "plot_graph",
    srcs = ["plot_graph.py"],
    main = "plot_graph.py",
    deps = [
        ":tf_proto_to_json",
        ":create_flask_app",
    ],
)

py_library(
    name = "tf_proto_to_json",
    srcs = ["tf_proto_to_json.py"],
    deps = [
    ],
)

py_library(
    name = "create_flask_app",
    srcs = ["create_flask_app.py"],
    data = [
        "//graph_visualization/templates:d3_graph",

    ],
)