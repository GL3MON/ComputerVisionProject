python - <<'PY'
import os
import vllm

base = os.path.dirname(vllm.__file__)
interface = os.path.join(base, "platforms", "interface.py")
cuda = os.path.join(base, "platforms", "cuda.py")

# Restore the original vLLM 0.17.0 interface implementation
with open(interface) as f:
    s = f.read()

start = s.index("    def device_id_to_physical_device_id(cls, device_id: int):")
start = s.rfind("    @classmethod", 0, start)

end = s.index("\n    @classmethod", start + 20)

original = '''    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int):
        # Treat empty device control env var as unset. This is a valid
        # configuration in Ray setups where the engine is launched in
        # a CPU-only placement group located on a GPU node.
        if (
            cls.device_control_env_var in os.environ
            and os.environ[cls.device_control_env_var] != ""
        ):
            device_ids = os.environ[cls.device_control_env_var].split(",")
            physical_device_id = device_ids[device_id]
            return int(physical_device_id)
        else:
            return device_id
'''

s = s[:start] + original + s[end:]
with open(interface, "w") as f:
    f.write(s)

# Add the MIG UUID resolver to NvmlCudaPlatform
with open(cuda) as f:
    s = f.read()

marker = "class NvmlCudaPlatform(CudaPlatformBase):"

if "def device_control_id_to_physical_device_id" in s:
    print("MIG resolver already exists in cuda.py")
else:
    patch = '''class NvmlCudaPlatform(CudaPlatformBase):
    @classmethod
    @with_nvml_context
    def device_control_id_to_physical_device_id(cls, device_id: str) -> int:
        try:
            return int(device_id)
        except ValueError:
            handle = pynvml.nvmlDeviceGetHandleByUUID(device_id)
            return pynvml.nvmlDeviceGetIndex(handle)
'''

    assert marker in s
    s = s.replace(marker, patch, 1)

    with open(cuda, "w") as f:
        f.write(s)

print("Patched vLLM 0.17.0 for MIG UUIDs")
print("interface:", interface)
print("cuda:", cuda)
PY


python - <<'PY'
import os
import vllm

path = os.path.join(
    os.path.dirname(vllm.__file__),
    "platforms",
    "interface.py",
)

with open(path) as f:
    s = f.read()

old = """    def device_id_to_physical_device_id(cls, device_id: int):
        # Treat empty device control env var as unset. This is a valid
        # configuration in Ray setups where the engine is launched in
        # a CPU-only placement group located on a GPU node.
        if (
            cls.device_control_env_var in os.environ
            and os.environ[cls.device_control_env_var] != ""
        ):
            device_ids = os.environ[cls.device_control_env_var].split(",")
            physical_device_id = device_ids[device_id]
            return int(physical_device_id)
        else:
            return device_id
"""

new = """    def device_id_to_physical_device_id(cls, device_id: int):
        # Treat empty device control env var as unset.
        if (
            cls.device_control_env_var in os.environ
            and os.environ[cls.device_control_env_var] != ""
        ):
            device_ids = os.environ[cls.device_control_env_var].split(",")
            physical_device_id = device_ids[device_id]

            try:
                return int(physical_device_id)
            except ValueError:
                import pynvml

                pynvml.nvmlInit()
                try:
                    handle = pynvml.nvmlDeviceGetHandleByUUID(
                        physical_device_id
                    )
                    return pynvml.nvmlDeviceGetIndex(handle)
                finally:
                    pynvml.nvmlShutdown()
        else:
            return device_id
"""

if old not in s:
    print("OLD FUNCTION NOT FOUND.")
    start = s.index("    def device_id_to_physical_device_id")
    print(s[start:start+700])
    raise SystemExit(1)

with open(path, "w") as f:
    f.write(s.replace(old, new))

print("PATCHED:", path)
PY