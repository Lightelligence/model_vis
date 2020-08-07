package(
    default_visibility = ["//visibility:public"],
)

load("//:rules.bzl", "lt_nodejs_library")

py_binary(
    name = "run_web_server",
    srcs = ["run_web_server.py"],
    data = [
        "//graph_visualization/templates:d3_graph",
    ],
    main = "run_web_server.py",
    deps = [
<<<<<<< HEAD
        ":tf_proto_to_json",
=======
        ":proto_to_json",
>>>>>>> fb77e5e39053806711e011258d2e7ff4348716bb
    ],
)

py_binary(
    name = "proto_to_json",
    srcs = ["proto_to_json.py"],
    main = "proto_to_json.py",
    deps = [
<<<<<<< HEAD
    ],
)

py_binary(
    name = "tf_proto_to_json",
    srcs = ["tf_proto_to_json.py"],
    main = "tf_proto_to_json.py",
    deps = [
=======
>>>>>>> fb77e5e39053806711e011258d2e7ff4348716bb
    ],
)
