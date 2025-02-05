#!/bin/bash

# This script runs the cyclogeostrophy impact experiment on NeurOST data, globally.
# Saving all the daily snapshot datasets requires around 1.5TB of disk space.

python main.py \
    ssh_data=local \
    ssh_data.resource.path='/summer/meom/workdir/bertrava/NEUROST_SSH-SST_L4_V2024.0.zarr' \
    drifter_data=local \
    drifter_data.resource.path='https://minio.dive.edito.eu/oidc-bertrava/data/gdp6h.zarr' \
    temporal_extent='["2010", "2023"]'
