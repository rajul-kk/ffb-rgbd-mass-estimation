"""Builds notebooks/kaggle_zeroshot_infer.ipynb: runs scripts/zeroshot_infer.py on Kaggle (GPU optional).

Before running on Kaggle: (1) upload the repo's ffb/ and scripts/ folders as one dataset, (2) upload
zeroshot_cache/inputs/*.npz (from scripts/zeroshot_export_inputs.py) as a second dataset, (3) edit the two paths
in the config cell, (4) turn Internet on (model downloads). Download /kaggle/working/masks.zip and unzip it to
zeroshot_cache/masks, then run scripts/zeroshot_score.py locally. The notebook is built, not executed here.
"""
import json
from pathlib import Path

cells = [
    ("markdown", "# Zero-shot mask inference (Grounding DINO, YOLO-World, SAM 2)\nSee docs/PROTOCOL_zeroshot.md. Produces raw masks only (the providers currently run on CPU torch even if a GPU is attached; porting to cuda is an open item); mass scoring runs locally with scripts/zeroshot_score.py."),
    ("code", "# Config: edit these two paths to match your uploaded datasets\nCODE_DIR = '/kaggle/input/ffb-code'      # contains ffb/ and scripts/\nINPUTS = '/kaggle/input/ffb-zeroshot-inputs'  # contains FFB*.npz\nOUT = '/kaggle/working/masks'"),
    ("code", "!pip install -q 'transformers>=4.56' ultralytics clip-anytorch ftfy regex opencv-python-headless scipy pandas"),
    ("code", "import os, shutil, sys\nshutil.copytree(CODE_DIR, '/kaggle/working/repo', dirs_exist_ok=True)\nos.chdir('/kaggle/working')  # ultralytics downloads yolov8s-worldv2.pt here\nsys.path.insert(0, '/kaggle/working/repo'); sys.path.insert(0, '/kaggle/working/repo/scripts')\nimport torch; print(torch.__version__, torch.cuda.is_available())"),
    ("code", "!python /kaggle/working/repo/scripts/zeroshot_infer.py --inputs $INPUTS --out $OUT"),
    ("code", "import shutil; shutil.make_archive('/kaggle/working/masks', 'zip', OUT); print(open(OUT + '/run_info.json').read())"),
]
nb = dict(nbformat=4, nbformat_minor=5, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          cells=[dict(cell_type=t, metadata={}, source=s.splitlines(True), **({"outputs": [], "execution_count": None} if t == "code" else {}))
                 for t, s in cells])
out = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_zeroshot_infer.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
