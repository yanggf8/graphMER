import csv
import json
from pathlib import Path
import unittest
from unittest.mock import patch, mock_open

from scripts.check_monitoring_gates import check_monitoring_gates
from scripts.update_metadata import update_ablation_metadata

class TestTpuTools(unittest.TestCase):

    def test_check_monitoring_gates_pass(self):
        """Test that the monitoring gates pass with sufficient improvement."""
        metrics_data = [
            ["step", "total_loss", "mlm_loss", "mnm_loss", "mlm_validation_accuracy", "mnm_validation_accuracy"],
            [1, 1.0, 1.0, 1.0, 0.1, 0.1],
            [10, 0.5, 0.5, 0.5, 0.2, 0.2],
            [20, 0.4, 0.4, 0.4, 0.3, 0.3],
        ]
        with patch("builtins.open", new_callable=mock_open, read_data=self.create_csv_string(metrics_data)):
            with patch("pathlib.Path.exists", return_value=True):
                success, message = check_monitoring_gates("dummy_metrics.csv")
                self.assertTrue(success)

    def test_check_monitoring_gates_fail_improvement(self):
        """Test that the monitoring gates fail with insufficient improvement."""
        metrics_data = [
            ["step", "total_loss", "mlm_loss", "mnm_loss", "mlm_validation_accuracy", "mnm_validation_accuracy"],
            [1, 1.0, 1.0, 1.0, 0.1, 0.1],
            [10, 0.9, 0.9, 0.9, 0.1, 0.105],
        ]
        with patch("builtins.open", new_callable=mock_open, read_data=self.create_csv_string(metrics_data)):
            success, message = check_monitoring_gates("dummy_metrics.csv")
            self.assertFalse(success)

    def test_check_monitoring_gates_fail_regression(self):
        """Test that the monitoring gates fail with loss regression."""
        metrics_data = [
            ["step", "total_loss", "mlm_loss", "mnm_loss", "mlm_validation_accuracy", "mnm_validation_accuracy"],
            [1, 0.5, 0.5, 0.5, 0.1, 0.1],
            [10, 1.0, 1.0, 1.0, 0.3, 0.3],
        ]
        with patch("builtins.open", new_callable=mock_open, read_data=self.create_csv_string(metrics_data)):
            success, message = check_monitoring_gates("dummy_metrics.csv")
            self.assertFalse(success)

    def test_update_ablation_metadata(self):
        """Test that the ablation metadata is updated correctly."""
        initial_metadata = {"schema_version": "1.0"}
        with patch("builtins.open", new_callable=mock_open, read_data=json.dumps(initial_metadata)) as m:
            with patch("json.dump") as mock_json_dump:
                update_ablation_metadata("config.yaml", 42, 100, "tpu", "tpu")
                # Check that the file was written to
                m.assert_called_with(Path("ablation_metadata.json"), 'w', encoding='utf-8')
                # Check that the correct data was written
                written_data = mock_json_dump.call_args[0][0]
                self.assertEqual(written_data["random_seed"], 42)
                self.assertEqual(written_data["run_parameters"]["steps"], 100)
                self.assertEqual(written_data["run_parameters"]["device_type"], "tpu")

    def create_csv_string(self, data):
        """Helper function to create a CSV string from a list of lists."""
        with self.subTest():
            from io import StringIO
            output = StringIO()
            writer = csv.writer(output)
            writer.writerows(data)
            return output.getvalue()

if __name__ == "__main__":
    unittest.main()
