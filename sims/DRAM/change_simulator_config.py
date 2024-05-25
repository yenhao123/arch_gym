import argparse

def modify_cfg_file(config_idx):
    # 读取 cfg 文件内容
    with open('/home/user/Desktop/oss-arch-gym/configs/sims/DRAMSys_config_sample.py', 'r') as file:
        filedata = file.read()
    
    new_data = filedata.replace('{config_idx}', str(config_idx))

    with open('/home/user/Desktop/oss-arch-gym/configs/sims/DRAMSys_config.py', 'w') as file:
        file.write(new_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify configuration file.')
    parser.add_argument('--config_idx', type=int, required=True, help='Configuration index to set in the config file')
    args = parser.parse_args()
    modify_cfg_file(args.config_idx)