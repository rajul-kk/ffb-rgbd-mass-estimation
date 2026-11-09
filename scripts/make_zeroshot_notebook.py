"""Builds notebooks/kaggle_zeroshot_infer.ipynb: runs scripts/zeroshot_infer.py on a Kaggle GPU.

Before running on Kaggle: (1) upload the repo's ffb/ and scripts/ folders and zeroshot_cache/inputs/*.npz
(from scripts/zeroshot_export_inputs.py) as one private dataset (paths are discovered), (2) attach a GPU and
turn Internet on (model downloads). Download /kaggle/working/masks.zip and unzip it to
zeroshot_cache/masks, then run scripts/zeroshot_score.py locally. The notebook is built, not executed here.
"""
import json
from pathlib import Path

INTRO = """# Zero-shot mask inference (Grounding DINO, YOLO-World, SAM 2)
See docs/PROTOCOL_zeroshot.md. Produces raw masks only, on the GPU (`--device cuda` fails loudly if none is attached); mass scoring runs locally with scripts/zeroshot_score.py."""

PATHS = """# Paths are discovered: attach one dataset holding ffb/, scripts/ and the FFB*.npz inputs (any layout)
import glob, os
INFER = glob.glob('/kaggle/input/**/scripts/zeroshot_infer.py', recursive=True)[0]
CODE_DIR = os.path.dirname(os.path.dirname(INFER))
INPUTS = os.path.dirname(sorted(glob.glob('/kaggle/input/**/FFB*.npz', recursive=True))[0])
OUT = '/kaggle/working/masks'
print(CODE_DIR, INPUTS)"""

INSTALL = "!pip install -q 'transformers>=4.56' ultralytics clip-anytorch ftfy regex opencv-python-headless scipy pandas"

SETUP = """import shutil, sys
shutil.copytree(CODE_DIR, '/kaggle/working/repo', dirs_exist_ok=True)
os.chdir('/kaggle/working')  # ultralytics downloads yolov8s-worldv2.pt here
sys.path.insert(0, '/kaggle/working/repo'); sys.path.insert(0, '/kaggle/working/repo/scripts')
import torch; print(torch.__version__, torch.cuda.is_available())"""

RUN = "!python /kaggle/working/repo/scripts/zeroshot_infer.py --inputs $INPUTS --out $OUT --device cuda"

ZIP = "import shutil; shutil.make_archive('/kaggle/working/masks', 'zip', OUT); print(open(OUT + '/run_info.json').read())"

cells = [("markdown", INTRO), ("code", PATHS), ("code", INSTALL), ("code", SETUP), ("code", RUN), ("code", ZIP)]
nb = dict(nbformat=4, nbformat_minor=5, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          cells=[dict(cell_type=t, metadata={}, source=s.splitlines(True), **({"outputs": [], "execution_count": None} if t == "code" else {}))
                 for t, s in cells])
out = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_zeroshot_infer.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
