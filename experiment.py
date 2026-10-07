from pathlib import Path
from digits_workflow import run

if __name__ == "__main__":
    run(Path(__file__).resolve().parent / "results")
