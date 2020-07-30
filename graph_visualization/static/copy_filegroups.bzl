"""
Custom rule to copy files from a list of filegroups into the current directory
Modified the code found here:
https://stackoverflow.com/questions/38905256/bazel-copy-multiple-files-to-binary-directory
"""

def _copy_filegroups_impl(ctx):
    all_input_files = [
        f
        for t in ctx.attr.filegroups
        for f in t.files
    ]

    all_outputs = []
    for f in all_input_files:
        out = ctx.actions.declare_file(f.basename)
        all_outputs += [out]
        ctx.actions.run_shell(
            outputs = [out],
            inputs = depset([f]),
            arguments = [f.path, out.path],
            command = "cp $1 $2",
        )

    if len(all_input_files) != len(all_outputs):
        fail("Output count should be 1-to-1 with input count.")

    return [
        DefaultInfo(
            files = depset(all_outputs),
            runfiles = ctx.runfiles(files = all_outputs),
        ),
    ]

copy_filegroups = rule(
    implementation = _copy_filegroups_impl,
    attrs = {
        "filegroups": attr.label_list(),
    },
)
