import collections
import os

import numpy as np

import proto_to_json
from sdk2.common import py_test_util
from sdk2.graph.tests import tf_mnist_sample
from sdk2.graph.transform_graph.graph_transformers.tests import \
    test_collapse_supported_subgraphs
from sdk2.proto import calibration_pb2


class test_d3_graphs(py_test_util.PythonTestCase):

    def _import_sample_graph(self):
        return tf_mnist_sample.generate_inference_light_graph()

    def test_d3_graph(self):
        light_graph = self._import_sample_graph()

        # Graph without subgraph
        proto_to_json.main(light_graph)

    def test_graph_with_ctrl_inputs(self):
        light_graph_with_ctrl_inputs =\
             test_collapse_supported_subgraphs.TestCollapseSupportedSubgraphs.\
             test_subgraph_control_inputs()

        proto_to_json.main(light_graph_with_ctrl_inputs)

    def test_subgraph(self):
        """Simple test for a graph that has a connected supported
        section with an unsupported node, simple assumptions about
        subgraph partitioning will fail here"""

        collapsed_graph_with_subgraph =\
            test_collapse_supported_subgraphs.TestCollapseSupportedSubgraphs.\
            create_graph_with_unsupported_path()

        dict_format_graph = proto_to_json.main(collapsed_graph_with_subgraph)

        self.assertEquals(
            len([
                per_node_dict["name"]
                for per_node_dict in dict_format_graph["nodes"]
                if not per_node_dict["name"].startswith("Dummy")
            ]), 4)

    def test_folder_path(self):
        histogram_folder_path = os.path.join(self.tmp_dir, "histograms")
        os.mkdir(histogram_folder_path)
        name_to_hist_dict = collections.defaultdict(list)
        for ctr in range(10):
            array = np.random.rand(300, 400)
            max_value = np.max(np.abs(array))
            num_bins = 1 << 12
            hist, bin_edges = np.histogram(array,
                                           bins=num_bins,
                                           range=(-max_value, max_value))
            cal_hist = self.create_empty_cal_hist_pb()
            cal_hist.hist.extend(hist)
            cal_hist.value[
                calibration_pb2.CalibrationHistogram.HISTOGRAM_MAX] = max_value
            filename = os.path.join(histogram_folder_path, "{}.pb".format(ctr))
            with open(filename, "wb") as f:
                f.write(cal_hist.SerializeToString())
        proto_to_json.fetch_all_histograms([histogram_folder_path], name_to_hist_dict)
        self.assertTrue(len(name_to_hist_dict.keys()) > 0)

    def create_empty_cal_hist_pb(self):
        cal_hist_pb = calibration_pb2.CalibrationHistogram()
        num_vals = len(calibration_pb2.CalibrationHistogram.ValueType.items())
        cal_hist_pb.value.extend([0] * num_vals)
        return cal_hist_pb


if __name__ == "__main__":
    py_test_util.main()
