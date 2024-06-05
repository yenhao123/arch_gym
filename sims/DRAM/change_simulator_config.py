import argparse

def modify_cfg_file(config_idx, workload_name):
    # 读取 cfg 文件内容
    with open(f'/home/user/Desktop/oss-arch-gym/configs/sims/DRAMSys_config_sample_{workload_name}.py', 'r') as file:
        filedata = file.read()
    
    new_data = filedata.replace('{config_idx}', str(config_idx))

    with open('/home/user/Desktop/oss-arch-gym/configs/sims/DRAMSys_config.py', 'w') as file:
        file.write(new_data)

def modify_cfg_file_all(workload_name):
    # 读取 cfg 文件内容
    with open(f'/home/user/Desktop/oss-arch-gym/configs/sims/DRAMSys_config_sample_{workload_name}.py', 'r') as file:
        filedata = file.read()
    
    new_data = filedata.replace('{config_idx}', "all")

    with open('/home/user/Desktop/oss-arch-gym/configs/sims/DRAMSys_config.py', 'w') as file:
        file.write(new_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify configuration file.')
    parser.add_argument('--config_idx', type=int, help='Configuration index to set in the config file')
    parser.add_argument('--is_all', action='store_true', help='Configuration index to set in the config file')
    args = parser.parse_args()
    workload_name = "blackscholes"
    if args.is_all:
        modify_cfg_file_all(workload_name)
    else:
        modify_cfg_file(args.config_idx, workload_name)