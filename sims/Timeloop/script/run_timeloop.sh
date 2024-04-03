#!/usr/bin/sh
#HOME="/home/workspace"

#cd /home/workspace/src/timeloop-examples/workspace/final-project/example_designs/eyeriss_like

OUTPUT_DIR="./oss-arch-gym/sims/Timeloop/output"
LAYER_SHAPE="AlexNet/AlexNet_layer1.yaml"

# Invoke timeloop
echo " " | timeloop-mapper ./oss-arch-gym/sims/Timeloop/arch/eyeriss_like.yaml \
./oss-arch-gym/sims/Timeloop/arch/components/*.yaml \
./oss-arch-gym/sims/Timeloop/mapper/mapper.yaml constraints/*.yaml \
../../layer_shapes/$LAYER_SHAPE >$OUTPUT_DIR/timeloop_simulation_output.txt

mv timeloop-mapper.stats.txt ./oss-arch-gym/sims/Timeloop/output
