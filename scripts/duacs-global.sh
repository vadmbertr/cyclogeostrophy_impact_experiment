#!/bin/bash

# This script runs the cyclogeostrophy impact experiment on DUACS data, globally.
# Saving all the daily snapshot datasets requires around 2TB of disk space.

python main.py \
    ssh_data=copernicus \
    ssh_data.resource.cms_dataset_id='cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D' \
    drifter_data=local \
    drifter_data.resource.path='https://minio.dive.edito.eu/oidc-bertrava/data/gdp6h.zarr' \
    temporal_extent='["2000", "2023"]'
