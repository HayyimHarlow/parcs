import unittest

from parcs_v4.assoc_operator_v4 import TieredMatchPolicy, PARCSv4Operator
from parcs_v4.metrics_v4 import compute_identity_stability
from parcs_v4.adversarial_v4 import FullAdversarialGenerator
from parcs_v4.type_confusion_v4 import TypeConfusionEvaluator
from parcs_v4.forward_model import ForwardSceneGenerator
from parcs_v4.configs import PARCSConfig


class TestPARCSV4(unittest.TestCase):

    def setUp(self):
        self.cfg = PARCSConfig()
        self.scene = ForwardSceneGenerator(self.cfg).generate_scene()
        self.adv_gen = FullAdversarialGenerator(self.cfg)
        self.adv_scene = self.adv_gen.apply_full_adversarial_pipeline(self.scene)
        self.policy = TieredMatchPolicy(self.cfg)
        self.operator = PARCSv4Operator(self.cfg, policy=self.policy)
        self.results = self.operator.run(self.adv_scene)

    def test_identity_stability_defined(self):
        stability = compute_identity_stability(self.results)
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)

    def test_detection_metrics_non_negative(self):
        from parcs_v4.metrics_v4 import compute_detection_metrics
        tp, fp, fn = compute_detection_metrics(self.results)
        self.assertGreaterEqual(tp, 0)
        self.assertGreaterEqual(fp, 0)
        self.assertGreaterEqual(fn, 0)

    def test_type_confusion_indexing(self):
        evaluator = TypeConfusionEvaluator(self.cfg)
        cm = evaluator.evaluate(self.results)
        # basic sanity: confusion matrix should not be all None
        self.assertTrue(cm is not None)

    def test_full_adversarial_pipeline_runs(self):
        # ensure adversarial pipeline does not break structure
        self.assertTrue(hasattr(self.adv_scene, "frames"))
        self.assertGreater(len(self.adv_scene.frames), 0)


if __name__ == "__main__":
    unittest.main()
