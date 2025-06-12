import copy
from ortools.sat.python import cp_model
import numpy as np
from schedule import Schedule

class CPSolver:
    def __init__(self):
        self.schedule = None
    
    def solve_blocking_job_shop(self,originData,mask=None,random_seed=2025,time_limit=300):
        data = copy.deepcopy(originData)
        processing_times = originData[0]
        machine_sequences = originData[1]
        num_jobs = len(processing_times)
        num_machines = len(machine_sequences[0])
        if mask is None:
            mask = np.zeros((num_jobs, num_machines))
            # 0 表示工序未完成，-1 表示已经完成  其他正整数表示工序剩余的时间（有些工序仅仅进行了一半）
        
        self.schedule = Schedule(num_jobs, num_machines)
        model = cp_model.CpModel()
        horizon = sum(sum(pt) for pt in processing_times)
        start_times = {}
        for j in range(num_jobs):
            for i in range(len(machine_sequences[j])):
                start_times[j, i] = model.NewIntVar(0, horizon, f'start_{j}_{i}')

        # 修改部分数据为已完成或进行中的状态
        for i in range(num_jobs):
            for j in range(num_machines):
                if mask[i][j] == -1:
                    model.Add(start_times[i,j]  == 0)
                    data[0][i][j] = 0
                    data[1][i][j] = 0

                elif mask[i][j] > 0:
                    model.Add(start_times[i,j]  == 0)
                    data[0][i][j] = mask[i][j]
        processing_times = data[0]
        machine_sequences = data[1]

        makespan = model.NewIntVar(0, 10*horizon, 'makespan')

        for j in range(num_jobs):
            for i in range(len(machine_sequences[j]) - 1):
                model.Add(start_times[j, i] + processing_times[j][i] <= start_times[j, i + 1])

        m = len(machine_sequences[0])
        for i1 in range(num_jobs):
            for i2 in range(num_jobs):
                for j1 in range(m):
                    for j2 in range(m):
                        if i1 == i2:
                            continue
                        if machine_sequences[i1][j1] == 0 or machine_sequences[i2][j2] == 0:
                            continue
                        if machine_sequences[i1][j1] == machine_sequences[i2][j2]:
                            b1 = model.NewBoolVar(f'b_{i1}_{j1}_{i2}_{j2}_1')
                            b2 = model.NewBoolVar(f'b_{i1}_{j1}_{i2}_{j2}_2')
                            model.Add(start_times[i1, j1] + processing_times[i1][j1] <= start_times[i2, j2]).OnlyEnforceIf(b1)
                            model.Add(start_times[i2, j2] + processing_times[i2][j2] <= start_times[i1, j1]).OnlyEnforceIf(b2)
                            model.AddBoolOr([b1,b2])
                        else :
                            continue
                        if j1 != m-1 and j2 != m-1 :
                            b3 = model.NewBoolVar(f'b_{i1}_{j1}_{i2}_{j2}_3')
                            b4 = model.NewBoolVar(f'b_{i1}_{j1}_{i2}_{j2}_4')
                            model.Add(start_times[i1, j1+1] <= start_times[i2, j2]).OnlyEnforceIf(b3)
                            model.Add(start_times[i2, j2+1]  <= start_times[i1, j1]).OnlyEnforceIf(b4)
                            model.AddBoolOr([b3,b4])
                        elif j2 != m-1:
                            b5 = model.NewBoolVar(f'b_{i1}_{j1}_{i2}_{j2}_5')
                            b6 = model.NewBoolVar(f'b_{i1}_{j1}_{i2}_{j2}_6')
                            model.Add(start_times[i2,j2] >= start_times[i1,m-1] + processing_times[i1][m-1] ).OnlyEnforceIf(b5)
                            model.Add(start_times[i1,m-1] >= start_times[i2,j2+1]).OnlyEnforceIf(b6)
                            model.AddBoolOr([b5,b6])
        for j in range(num_jobs):
            last_op = len(machine_sequences[j]) - 1
            model.Add(start_times[j, last_op] + processing_times[j][last_op] <= makespan)
        
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        # 设置最大运行时间（秒）
        solver.parameters.max_time_in_seconds = time_limit
        # 设置随机种子
        solver.parameters.random_seed = random_seed
        # 设置日志级别
        #solver.parameters.log_search_progress = True
        
        status = solver.Solve(model)
        #print(solver.Value(makespan))
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for j in range(num_jobs):
                for i, m in enumerate(machine_sequences[j]):
                    self.schedule.add_record(j,i,solver.Value(start_times[j,i]), solver.Value(start_times[j,i])+ processing_times[j][i])
        else:
            print('No feasible solution found !!!!!!!!!!')
        return self.schedule
        
if __name__ == '__main__':
    data = np.load('benchmark/la/la01.npy',allow_pickle=True)
    data = data[0]
    data = [[[ 0 , 0 , 0 , 6 ,37], [ 0 , 0 ,31 ,55 ,34], [ 0 , 0  ,0, 28, 62], [ 0 , 0 , 0 ,18 ,12], [77 ,79 ,43 ,75 ,96]],
            [[0, 0, 0 ,2 ,5], [0, 0, 5, 4 ,3], [0, 0, 0 ,1 ,4], [0, 0, 0, 3 ,1], [5, 4, 3 ,2 ,1]]]
    mask = [[-1. ,-1. ,-1.,  6. , 0.], [-1., -1., 31.,  0. , 0.], [-1. ,-1., -1., 28. , 0.], [-1. ,-1., -1. ,18. , 0.], [ 0.,  0. , 0.,  0.,  0.]]
    print(data)
    cp_solver = CPSolver()
    sc = cp_solver.solve_blocking_job_shop(data,mask)
    sc.plotSchedule(data)
    exit(0)

    solver = CPSolver()
    schedule = solver.solve_blocking_job_shop(data)
    schedule.plotSchedule(data)