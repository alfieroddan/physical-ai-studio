from physicalai.inference import InferenceModel
from physicalai.policies import Pi05

REPO_ID = "lerobot/pi05_libero_base"
EXPORT_DIR = "./exports/pi05_libero"

# load policy
print(f"[pi05] loading policy from {REPO_ID}")
policy = Pi05(pretrained_name_or_path=REPO_ID)
print("[pi05] policy loaded, switching to eval()")
policy.eval()
print(f"[pi05] exporting torch model to {EXPORT_DIR}")
policy.export(EXPORT_DIR, backend="torch")

# load
print(f"[pi05] loading exported torch inference model from {EXPORT_DIR}")
model = InferenceModel.load(EXPORT_DIR)
print("[pi05] torch export round-trip complete")

# openvino version

REPO_ID = "lerobot/pi05_libero_base"
EXPORT_DIR = "./exports/pi05_libero"

# load policy
print(f"[pi05] loading policy from {REPO_ID} for openvino export")
policy = Pi05(pretrained_name_or_path=REPO_ID)
print("[pi05] policy loaded, switching to eval()")
policy.eval()
print(f"[pi05] exporting openvino model to {EXPORT_DIR}")
policy.export(EXPORT_DIR, backend="openvino")

# load
print(f"[pi05] loading exported openvino inference model from {EXPORT_DIR}")
model = InferenceModel.load(EXPORT_DIR)
print("[pi05] openvino export round-trip complete")


REPO_ID = "lerobot/pi05_base"
EXPORT_DIR = "./exports/pi05_base"

# load policy
print(f"[pi05] loading policy from {REPO_ID}")
policy = Pi05(pretrained_name_or_path=REPO_ID)
print("[pi05] policy loaded, switching to eval()")
policy.eval()
print(f"[pi05] exporting torch model to {EXPORT_DIR}")
policy.export(EXPORT_DIR, backend="torch")

# load
print(f"[pi05] loading exported torch inference model from {EXPORT_DIR}")
model = InferenceModel.load(EXPORT_DIR)
print("[pi05] torch export round-trip complete")

# openvino version

REPO_ID = "lerobot/pi05_base"
EXPORT_DIR = "./exports/pi05_base"

# load policy
print(f"[pi05] loading policy from {REPO_ID} for openvino export")
policy = Pi05(pretrained_name_or_path=REPO_ID)
print("[pi05] policy loaded, switching to eval()")
policy.eval()
print(f"[pi05] exporting openvino model to {EXPORT_DIR}")
policy.export(EXPORT_DIR, backend="openvino")

# load
print(f"[pi05] loading exported openvino inference model from {EXPORT_DIR}")
model = InferenceModel.load(EXPORT_DIR)
print("[pi05] openvino export round-trip complete")
