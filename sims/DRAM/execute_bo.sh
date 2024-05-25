#!/bin/bash

for i in {0..9}
do
    python change_simulator_config.py --config_idx $i
    python train_bo_DRAMSys.py --config_idx $i
done
