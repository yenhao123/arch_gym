
def modify_cfg_file_canny(i):
    # 读取 cfg 文件内容
    with open('/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ddr4-example-canny.json', 'r') as file:
        filedata = file.read()

    # 替换占位符 {input1} 为实际的值
    new_data = filedata.replace('{workload_name}', f"dramsys_input_{i}")

    # 将修改后的内容写回 cfg 文件
    with open('/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/canny/ddr4-{}.json'.format(i), 'w') as file:
        file.write(new_data)

def modify_cfg_file_blackscholes(i):

    # 读取 cfg 文件内容
    with open('/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ddr4-example-blackscholes.json', 'r') as file:
        filedata = file.read()

    # 替换占位符 {input1} 为实际的值
    new_data = filedata.replace('{workload_name}', f"dramsys_input_{i}")

    # 将修改后的内容写回 cfg 文件
    with open('/home/user/Desktop/oss-arch-gym/sims/DRAM/DRAMSys/library/simulations/ours/blackscholes/ddr4-{}.json'.format(i), 'w') as file:
        file.write(new_data)

def main():
    import os
    os.makedirs("output", exist_ok=True)
    workload = "blackscholes"
    if workload == "canny":
        n_groups = 10
        for i in range(n_groups):
            modify_cfg_file_canny(i)
            
        modify_cfg_file_canny("all")
    elif workload == "blackscholes":
        n_groups = 10
        for i in range(n_groups):
            modify_cfg_file_blackscholes(i)
            
        modify_cfg_file_blackscholes("all")

if __name__ == "__main__":
    main()