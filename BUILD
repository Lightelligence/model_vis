load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "setup",
    srcs = ["setup.py"],
    data = [
        "MANIFEST.in",
        "README.md",
    ],
)

sh_binary(
    name = "build_wheel",
    srcs = ["build_wheel.sh"],
    data = [
        ":setup",
        "//graph_visualization:plot_graph",
    ],
)
