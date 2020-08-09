import tf_proto_to_json
from sdk2.common import py_test_util


class test_tf_d3_graph(py_test_util.PythonTestCase):

    def _import_sample_tf_graph(self):
        pass

    def test_tf_d3_graph(self):
        graph = ""
        tf_proto_to_json.main(graph)

    def test_tf_graph_with_ctrl_inputs(self):
        graph = ""
        tf_proto_to_json(graph)

    def test_folder_path(self, path):
        pass


if __name__ == "__main__":
    py_test_util.main()
