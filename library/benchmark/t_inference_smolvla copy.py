from physicalai.inference import InferenceModel
from physicalai.policies import SmolVLA

EXPORT_DIR = "./tmp-benchmark/tmp/smolvla_libero_torch"

# load policy
policy = SmolVLA(pretrained_name_or_path="lerobot/smolvla_libero")
policy.eval()
print(policy.get_supported_export_backends())
policy.export(EXPORT_DIR, backend="torch")

# benchmark
model = InferenceModel.load(EXPORT_DIR)

# # benchmark
# benchmark = InferenceLatencyBenchmark(max_iters=50, warmup_iters=50)

# def make_inputs(seed: int = 0):
# 	rng = np.random.default_rng(seed)
# 	while True:
# 		yield {
# 			"task": ["pick up the red block"],
# 			"state": rng.standard_normal((1, 8), dtype=np.float32),
# 			"images": {
# 				"image": rng.standard_normal((1, *CAMERA_SHAPE), dtype=np.float32),
# 				"image2": rng.standard_normal((1, *CAMERA_SHAPE), dtype=np.float32),
# 			},
# 		}


# inputs = make_inputs(seed=0)

# metrics = benchmark.run(model, inputs)

# print("Latency metrics:", metrics)
# print(
# 	"Latency (ms): "
# 	f"warmup_avg={metrics['avg_warmup_iter_time'] * 1000:.2f}, "
# 	f"mean={metrics['mean_iter_time'] * 1000:.2f}, "
# 	f"median={metrics['median_iter_time'] * 1000:.2f}, "
# 	f"min={metrics['min_iter_time'] * 1000:.2f}, "
# 	f"max={metrics['max_iter_time'] * 1000:.2f}, "
# 	f"std={metrics['std_iter_time'] * 1000:.2f}"
# )
