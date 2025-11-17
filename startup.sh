#!/bin/bash

# Install dependencies
pip install gdown

# Run download scripts
python download_ckpts.py
python download_data.py

# Start JupyterLab in foreground (keeps container alive)
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token='' --ServerApp.password='' --ServerApp.disable_check_xsrf=True --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True
