# ProjectChippy
Project utilizing Raspberry Pi to detect squirrels and dispensing the nuts remotely

## Installation

Create and activate a virtual environment that can see the system Python packages:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
```

Install the project and its dependencies in editable mode from the repository root:

```bash
source .venv/bin/activate
pip install -e .
```

## Run the project

From the repository root run the package entry point:

```bash
python -m project_chippy
```

The trained model file should be placed at `models/model.onnx` before running the project. In the future, this model will be pulled from the cloud automatically. The current model contains squirrel, wood pigeon, and pigeon labels.
