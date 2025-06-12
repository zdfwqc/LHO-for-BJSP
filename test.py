from LHOSolver import LHOSolver
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    data = np.load('benchmark/la/la11.npy',allow_pickle=True)
    data = data[0]
    solver = LHOSolver(numofMachines=5,k = 7)
    solver.reset(data)
    solver.loadNet('model/dsh75_NORM.pth')
    net_results = []
    greedy_results = []
    net_results.append(solver.solve(data, model='net'))
    for _ in range(10):  # 运行10次取平均

        greedy_results.append(solver.solve(data, model='greedy'))
    
    print("神经网络模型平均结果:", np.mean(net_results))
    print("贪心算法平均结果:", np.mean(greedy_results))
