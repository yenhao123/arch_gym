#!/usr/bin/env python3

import subprocess
import os
import numpy as np
import yaml


class TimeloopWrapper:
    def __init__(self, script_dir=None, output_dir=None, arch_dir=None, mapper_dir=None, workload_dir=None):
        self.script_dir   = script_dir
        self.output_dir   = output_dir
        self.arch_dir     = arch_dir
        self.mapper_dir   = mapper_dir
        self.workload_dir = workload_dir
        return


    def prepare_cmd(self):
        eyeriss_dir = ':/home/workspace/src/timeloop-examples/workspace/final-project/example_designs/eyeriss_like/'
        bind_script = self.script_dir + eyeriss_dir + 'script'                          # Entry point script
        bind_output = self.output_dir + eyeriss_dir + 'output'                          # Output directory
        bind_arch   = self.arch_dir   + eyeriss_dir + 'arch'                            # Arch directory
        bind_mapper = self.mapper_dir + eyeriss_dir + 'mapper'                          # Mapper directory
        cmd = ['singularity', 'run', '--writable-tmpfs', '--bind', 
                '{},{},{},{}'.format(bind_script, bind_output, bind_arch, bind_mapper),
               'timeloop_4_archgym']
        return cmd
    
    
    def launch_timeloop(self):
        energy = np.float64()
        area   = np.float64()
        cycles = np.float64()
        for layer in os.listdir(self.workload_dir): 
            self.modify_script(self.output_dir, layer)        
            cmd = self.prepare_cmd()
            completed = subprocess.run(cmd)
            mapping_exists = self.valid_mapping()
            if not mapping_exists: 
                energy, area, cycles = (-1.0, -1.0, -1.0)
                break
            metrics = self.obtain_metrics()
            energy += metrics[0]
            area    = metrics[1]  # Area does not change based on layer
            cycles += metrics[2]
            
        return energy, area, cycles


    def modify_script(self, output_dir, layer):
        # Update layer and output dir in run_timeloop script 
        script = "run_timeloop.sh"
        file = open(self.script_dir + "/" + script, "r")
        replacement = ""
        for line in file:
            line = line.strip()
            # if 'OUTPUT_DIR=' in line: 
            #     changes = 'OUTPUT_DIR=' + '"./' + output_dir.split('/')[-1] + '"'
            #     replacement = replacement + changes + "\n"
            if 'LAYER_SHAPE=' in line: 
                changes = 'LAYER_SHAPE=' + '"' + self.workload_dir.split('/')[-1] + '/' + layer + '"' 
                replacement = replacement + changes + "\n"
            else:
                replacement = replacement + line + "\n"
    
        file.close()
        fout = open(self.script_dir + "/" + script, "w")
        fout.write(replacement)
        fout.close()  
        print(layer)
        return 


    def valid_mapping(self):
        file = open(self.output_dir + "/timeloop_simulation_output.txt", "r")
        for line in file:
            line = line.strip()
            if "Summary stats for best mapping found by mapper:" in line:
                return True
        return False


    def obtain_metrics(self):
        file = open(self.output_dir + "/timeloop-mapper.stats.txt", "r")
        energy = np.float64()
        area   = np.float64()
        cycles = np.float64()
        for line in file: 
            line = line.strip()
            if "Area:" in line:
                area = float(line.split(' ')[1])
            elif "Energy:" in line:
                energy = float(line.split(' ')[1])
            elif "Cycles:" in line:
                cycles = float(line.split(' ')[1])
            
        return energy, area, cycles 


    def update_arch(self, arch_params):
        file = open(self.arch_dir + "/eyeriss_like.yaml", 'r')
        eyeriss_arch = yaml.safe_load(file)
    
        # CONSTANT POINTERS TO ARCH PARAMS
        PTRS = {
        'SHARED_GLB_CLASS'       : ['architecture','subtree',0,'subtree',0,'local',0,'class'],
        'SHARED_GLB_ATTRIBUTES'  : ['architecture','subtree',0,'subtree',0,'local',0,'attributes'],
        'DUMMY_BUFFER_CLASS'     : ['architecture','subtree',0,'subtree',0,'local',1,'class'],
        'DUMMY_BUFFER_ATTRIBUTES': ['architecture','subtree',0,'subtree',0,'local',1,'attributes'],
        'NUM_PEs'                : ['architecture','subtree',0,'subtree',0,'subtree',0,'name'],
        'IFMAP_SPAD_CLASS'       : ['architecture','subtree',0,'subtree',0,'subtree',0,'local',0,'class'],
        'IFMAP_SPAD_ATRIBUTES'   : ['architecture','subtree',0,'subtree',0,'subtree',0,'local',0,'attributes'],
        'WEIGHTS_SPAD_CLASS'     : ['architecture','subtree',0,'subtree',0,'subtree',0,'local',1,'class'],
        'WEIGHTS_SPAD_ATRIBUTES' : ['architecture','subtree',0,'subtree',0,'subtree',0,'local',1,'attributes'],
        'PSUM_SPAD_CLASS'        : ['architecture','subtree',0,'subtree',0,'subtree',0,'local',2,'class'],
        'PSUM_SPAD_ATRIBUTES'    : ['architecture','subtree',0,'subtree',0,'subtree',0,'local',2,'attributes'],
        'MAC_MESH_X'             : ['architecture','subtree',0,'subtree',0,'subtree',0,'local',3,'attributes','meshX']
        }
              
        # Update arch params passed in
        for key in arch_params.keys():
            ptr_list = PTRS[key]
            ptr = eyeriss_arch
            for level in ptr_list[:-1]:
                ptr = ptr[level] 
            ptr[ptr_list[-1]] = arch_params[key]
        
        # Write out updated eyeriss_like.yaml 
        file.close()
        fout = open(self.arch_dir + "/eyeriss_like.yaml", 'w')
        fout.write(yaml.dump(eyeriss_arch))
        fout.close()
        return   


    def get_arch_param_template(self):
        arch_params = {
                        'SHARED_GLB_CLASS'       : 'smartbuffer_SRAM',
                        'SHARED_GLB_ATTRIBUTES'  : {'memory_depth': 16384, 'memory_width': 64, 'n_banks': 32, 'block-size': 4, 
                                                    'word-bits': 16, 'read_bandwidth': 16, 'write_bandwidth': 16},
                        'DUMMY_BUFFER_CLASS'     : 'SRAM',
                        'DUMMY_BUFFER_ATTRIBUTES': {'depth': 16, 'width': 16, 'word-bits': 16, 'block-size': 1, 'meshX': 14},
                        'NUM_PEs'                : 'PE[0..167]',
                        'IFMAP_SPAD_CLASS'       : 'smartbuffer_SRAM',
                        'IFMAP_SPAD_ATRIBUTES'   : {'memory_depth': 12, 'memory_width': 16, 'block-size': 1, 'word-bits': 16, 
                                                    'meshX': 14, 'read_bandwidth': 2, 'write_bandwidth': 2},
                        'WEIGHTS_SPAD_CLASS'     : 'smartbuffer_SRAM',
                        'WEIGHTS_SPAD_ATRIBUTES' : {'memory_depth': 192, 'memory_width': 16, 'block-size': 1, 'word-bits': 16, 
                                                    'meshX': 14, 'read_bandwidth': 2, 'write_bandwidth': 2},
                        'PSUM_SPAD_CLASS'        : 'smartbuffer_SRAM',
                        'PSUM_SPAD_ATRIBUTES'    : {'memory_depth': 16, 'memory_width': 16, 'update_fifo_depth': 2, 'block-size': 1, 
                                                    'word-bits': 16, 'meshX': 14, 'read_bandwidth': 2, 'write_bandwidth': 2},
                        'MAC_MESH_X'             : 14
                      }
        return arch_params
    
    
#sudo docker run -v /home/sprakash/Documents/Repos/Timeloop/script:/home/workspace/src/timeloop-examples/workspace/final-project/example_designs/eyeriss_like/script -v /home/sprakash/Documents/Repos/Timeloop/output:/home/workspace/src/timeloop-examples/workspace/final-project/example_designs/eyeriss_like/output -v /home/sprakash/Documents/Repos/Timeloop/arch:/home/workspace/src/timeloop-examples/workspace/final-project/example_designs/eyeriss_like/arch -v /home/sprakash/Documents/Repos/Timeloop/mapper:/home/workspace/src/timeloop-examples/workspace/final-project/example_designs/eyeriss_like/mapper --user root -it timeloop_4_archgym 

