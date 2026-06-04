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


EXPORT_DIR = "./tmp-benchmark/tmp/smolvla_base_torch"
# load policy
policy = SmolVLA(pretrained_name_or_path="lerobot/smolvla_base")
policy.eval()
print(policy.get_supported_export_backends())
policy.export(EXPORT_DIR, backend="torch")
# benchmark
model = InferenceModel.load(EXPORT_DIR)
