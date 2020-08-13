import argparse
import os

import tf_proto_to_json
from flask import Flask, jsonify, render_template

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__),
                                         "templates"))


@app.route("/")
def home():
    """A template is rendered as Python flask searches for all html files
     in 'templates'folder, which is also why all
     HTML files are placed in that directory."""
    return render_template("d3_graph.html")


@app.route("/graph.json")
def graph():
    """ Serves a graph which is converted to JSON format."""
    return jsonify(app.config["jsonified"])


def main():
    """ Runs a web server based on flask, specifying port is optional
    (defaulted to 5000) """
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",
                        type=int,
                        help="port number (default 5000)",
                        default=5000)
    parser.add_argument("--pb_graph_path",
                        type=str,
                        help="path to a pb graph file")
    parser.add_argument(
        "--pb_histogram_folder_path",
        type=str,
        help="path to a histogram folder containing protobuf files",
        action="append")
    parser.add_argument("--reduce_unsupported",
                        action="store_true",
                        help=("reduces unsupported nodes by deleting them if"
                              "it is a unsupp -> unsupp connection"),
                        default=False),
    parser.add_argument("--reduce_while_nodes",
                        action="store_true",
                        help=("removes all '/while/' nodes by deleting them"
                              "before passing it over to json file"),
                        default=True)
    parser.add_argument(
        "--workload",
        type=str,
        help="workload name, must be a string in performance_sweep_map.py")
    parser.add_argument("--error_output_dir",
                        type=str,
                        help="Output directory for storing error pb")
    parser.add_argument("--no_text", default=False, action="store_true")
    parser.add_argument(
        "--graph_pb_type",
        default="tf",
        type=str,
        help=
        ("Currently defaulted to 'tf' , Other supported types in future releases are 'onnx' ,'lgf'"
         ))
    args = parser.parse_args()
    app.config["jsonified"] = tf_proto_to_json.main(
        args.pb_graph_path,
        graph_pb_type=args.graph_pb_type,
        reduce_unsupported=args.reduce_unsupported,
        reduce_while_nodes=args.reduce_while_nodes,
        list_of_pb_histogram_folder_paths=args.pb_histogram_folder_path,
        workload=args.workload,
        error_output_dir=args.error_output_dir)
    app.run(debug=True, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
