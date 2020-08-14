package(default_visibility = ["//visibility:public"])



filegroup(
    name = "manifest",
    srcs = [
        "MANIFEST.in",
    ],
)

py_library(
    name = "setup",
    srcs = ["setup.py"],
)

sh_binary(
    name = "build_wheel",
    srcs = ["build_wheel.sh"],
    data = [
        "//graph_visualization:plot_graph",
        ":manifest",
        ":setup"
    ],
)