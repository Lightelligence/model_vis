load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "extra_pip_files",
    srcs = [
        "MANIFEST.in",
        "README.md",
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
        ":extra_pip_files",
        ":setup",
        "//graph_visualization:plot_graph",
    ],
)
