import runpy
import pathlib
import sys

# Determine project root and module name
project_root = pathlib.Path(__file__).resolve().parent
pwd = pathlib.Path.cwd().resolve()

# Determine which .py file is open in the editor (via env var)
target_file = pathlib.Path(sys.argv[1]).resolve()

# Compute module path
rel = target_file.relative_to(project_root)
module = ".".join(rel.with_suffix("").parts)

print(f"Running module: {module}")
runpy.run_module(module, run_name="__main__")
