from LHOSolver import LHOSolver
from uniform_instance_gen import uni_instance_gen
from network.deepSubmudularFunc import DSFDeepSet
import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # 从文件加载数据或生成随机数据
    use_random = True # 设置是否使用随机生成的数据
    if use_random:
        num_instances = 100  # 生成10个随机实例
        all_net_results = []
        all_greedy_results = []
        all_random_results = []
        model =  DSFDeepSet()
        solver = LHOSolver(numofMachines=5, k=5,net=model)
        solver.loadNet('model/dsf75.pth')
        for i in range(num_instances):
            # 生成15个工件5个机器的随机实例,加工时间在1-99之间
            np.random.seed(i)
            times, machines = uni_instance_gen(60, 5, 1, 100)
            data = (times, machines)

            solver.reset(data)

            
            net_result = solver.solve(data, model='net')
            solver.reset(data)
            greedy_result = solver.solve(data, model='greedy')
            solver.reset(data)
            random_result = solver.solve(data, model='random')
            
            all_net_results.append(net_result)
            all_greedy_results.append(greedy_result)
            all_random_results.append(random_result)
            
            print(f"\n实例 {i+1} 的结果:")
            print(f"神经网络模型结果: {net_result}")
            print(f"贪心算法结果: {greedy_result}")
            print(f"随机算法结果: {random_result}")
            
        print("\n所有实例的平均结果:")
        print("神经网络模型平均结果:", np.mean(all_net_results))
        print("贪心算法平均结果:", np.mean(all_greedy_results)) 
        print("随机算法平均结果:", np.mean(all_random_results))
        
    else:
        data = np.load('benchmark/la/la01.npy',allow_pickle=True)
        data = data[0]
        model =  DSFDeepSet(input_dim=15)
        model.load_state_dict(torch.load('model/dsf75.pth'))
        solver = LHOSolver(numofMachines=5, k=5,net=model)
        # solver.loadNet('model/dsf75.pth')
        
        net_result = solver.solve(data, model='net')
        greedy_result = solver.solve(data, model='greedy')
        random_result = solver.solve(data, model='random')
        
        print("\n基准测试实例结果:")
        print("神经网络模型结果:", net_result)
        print("贪心算法结果:", greedy_result)
        print("随机算法结果:", random_result)
