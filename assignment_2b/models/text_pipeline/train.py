"""Train the text pipeline on every split and seed.

Run from assignment_2b/:  python -m models.text_pipeline.train [--splits ...] [--seeds ...]
See models/common/ for the shared blocks (data, features, nets, engine).
"""
from models.common.engine import cli

if __name__ == "__main__":
    cli("text", "train")
