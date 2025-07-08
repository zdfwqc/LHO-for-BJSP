import copy
from ortools.sat.python import cp_model
import numpy as np
from schedule import Schedule

class CPSolver:
    def __init__(self):
        self.schedule = None
    
    def solve_blocking_job_shop(self,originData,mask=None,random_seed=2025,time_limit=50,bws=True):
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
        horizon = int(sum(sum(pt) for pt in processing_times))
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
                            if bws:
                                model.Add(start_times[i1, j1] + processing_times[i1][j1] <= start_times[i2, j2]).OnlyEnforceIf(b1)
                                model.Add(start_times[i2, j2] + processing_times[i2][j2] <= start_times[i1, j1]).OnlyEnforceIf(b2)
                            else:
                                model.Add(start_times[i1, j1] + processing_times[i1][j1] <= start_times[i2, j2]).OnlyEnforceIf(b1)
                                model.Add(start_times[i2, j2] + processing_times[i2][j2] <= start_times[i1, j1]).OnlyEnforceIf(b2)
                            model.AddBoolOr([b1,b2])
                        else :
                            continue
                        if j1 != m-1 and j2 != m-1 :
                            b3 = model.NewBoolVar(f'b_{i1}_{j1}_{i2}_{j2}_3')
                            b4 = model.NewBoolVar(f'b_{i1}_{j1}_{i2}_{j2}_4')
                            if bws:
                                model.Add(start_times[i1, j1+1] <= start_times[i2, j2]).OnlyEnforceIf(b3)
                                model.Add(start_times[i2, j2+1]  <= start_times[i1, j1]).OnlyEnforceIf(b4)
                            else:
                                model.Add(start_times[i1, j1+1] < start_times[i2, j2]).OnlyEnforceIf(b3)
                                model.Add(start_times[i2, j2+1]  < start_times[i1, j1]).OnlyEnforceIf(b4)
                            model.AddBoolOr([b3,b4])
                        elif j2 != m-1:
                            b5 = model.NewBoolVar(f'b_{i1}_{j1}_{i2}_{j2}_5')
                            b6 = model.NewBoolVar(f'b_{i1}_{j1}_{i2}_{j2}_6')
                            if bws:
                                model.Add(start_times[i2,j2] >= start_times[i1,m-1] + processing_times[i1][m-1] ).OnlyEnforceIf(b5)
                                model.Add(start_times[i1,m-1] >= start_times[i2,j2+1]).OnlyEnforceIf(b6)
                            else:
                                model.Add(start_times[i2,j2] > start_times[i1,m-1] + processing_times[i1][m-1] ).OnlyEnforceIf(b5)
                                model.Add(start_times[i1,m-1] > start_times[i2,j2+1]).OnlyEnforceIf(b6)
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
    data = [[[21, 34, 56, 12, 78, 45, 89, 32, 67, 90, 43, 76, 23, 89, 45, 12, 65, 87, 34, 56],
             [43, 76, 23, 89, 45, 12, 65, 87, 34, 56, 67, 12, 45, 78, 34, 90, 23, 56, 89, 11],
             [67, 12, 45, 78, 34, 90, 23, 56, 89, 11, 89, 45, 67, 23, 56, 78, 11, 90, 12, 34],
             [89, 45, 67, 23, 56, 78, 11, 90, 12, 34, 12, 78, 90, 45, 23, 67, 34, 11, 56, 89],
             [12, 78, 90, 45, 23, 67, 34, 11, 56, 89, 56, 23, 11, 67, 89, 34, 78, 45, 90, 12],
             [56, 23, 11, 67, 89, 34, 78, 45, 90, 12, 34, 90, 78, 11, 12, 56, 45, 67, 23, 89],
             [34, 90, 78, 11, 12, 56, 45, 67, 23, 89, 78, 11, 34, 56, 90, 23, 67, 89, 45, 12],
             [78, 11, 34, 56, 90, 23, 67, 89, 45, 12, 45, 67, 89, 90, 11, 78, 12, 34, 56, 23],
             [45, 67, 89, 90, 11, 78, 12, 34, 56, 23, 90, 56, 12, 34, 67, 89, 23, 78, 11, 45],
             [90, 56, 12, 34, 67, 89, 23, 78, 11, 45, 21, 34, 56, 12, 78, 45, 89, 32, 67, 90],
             [21, 34, 56, 12, 78, 45, 89, 32, 67, 90, 43, 76, 23, 89, 45, 12, 65, 87, 34, 56],
             [43, 76, 23, 89, 45, 12, 65, 87, 34, 56, 67, 12, 45, 78, 34, 90, 23, 56, 89, 11],
             [67, 12, 45, 78, 34, 90, 23, 56, 89, 11, 89, 45, 67, 23, 56, 78, 11, 90, 12, 34],
             [89, 45, 67, 23, 56, 78, 11, 90, 12, 34, 12, 78, 90, 45, 23, 67, 34, 11, 56, 89],
             [12, 78, 90, 45, 23, 67, 34, 11, 56, 89, 56, 23, 11, 67, 89, 34, 78, 45, 90, 12],
             [56, 23, 11, 67, 89, 34, 78, 45, 90, 12, 34, 90, 78, 11, 12, 56, 45, 67, 23, 89],
             [34, 90, 78, 11, 12, 56, 45, 67, 23, 89, 78, 11, 34, 56, 90, 23, 67, 89, 45, 12],
             [78, 11, 34, 56, 90, 23, 67, 89, 45, 12, 45, 67, 89, 90, 11, 78, 12, 34, 56, 23],
             [45, 67, 89, 90, 11, 78, 12, 34, 56, 23, 90, 56, 12, 34, 67, 89, 23, 78, 11, 45],
             [90, 56, 12, 34, 67, 89, 23, 78, 11, 45, 21, 34, 56, 12, 78, 45, 89, 32, 67, 90]],
            [[3, 8, 15, 4, 19, 6, 12, 1, 17, 10, 5, 16, 2, 14, 7, 20, 11, 9, 13, 18],
             [15, 4, 19, 6, 12, 1, 17, 10, 5, 16, 2, 14, 7, 20, 11, 9, 13, 18, 3, 8],
             [12, 1, 17, 10, 5, 16, 2, 14, 7, 20, 11, 9, 13, 18, 3, 8, 15, 4, 19, 6],
             [5, 16, 2, 14, 7, 20, 11, 9, 13, 18, 3, 8, 15, 4, 19, 6, 12, 1, 17, 10],
             [7, 20, 11, 9, 13, 18, 3, 8, 15, 4, 19, 6, 12, 1, 17, 10, 5, 16, 2, 14],
             [13, 18, 3, 8, 15, 4, 19, 6, 12, 1, 17, 10, 5, 16, 2, 14, 7, 20, 11, 9],
             [15, 4, 19, 6, 12, 1, 17, 10, 5, 16, 2, 14, 7, 20, 11, 9, 13, 18, 3, 8],
             [12, 1, 17, 10, 5, 16, 2, 14, 7, 20, 11, 9, 13, 18, 3, 8, 15, 4, 19, 6],
             [5, 16, 2, 14, 7, 20, 11, 9, 13, 18, 3, 8, 15, 4, 19, 6, 12, 1, 17, 10],
             [7, 20, 11, 9, 13, 18, 3, 8, 15, 4, 19, 6, 12, 1, 17, 10, 5, 16, 2, 14],
             [13, 18, 3, 8, 15, 4, 19, 6, 12, 1, 17, 10, 5, 16, 2, 14, 7, 20, 11, 9],
             [15, 4, 19, 6, 12, 1, 17, 10, 5, 16, 2, 14, 7, 20, 11, 9, 13, 18, 3, 8],
             [12, 1, 17, 10, 5, 16, 2, 14, 7, 20, 11, 9, 13, 18, 3, 8, 15, 4, 19, 6],
             [5, 16, 2, 14, 7, 20, 11, 9, 13, 18, 3, 8, 15, 4, 19, 6, 12, 1, 17, 10],
             [7, 20, 11, 9, 13, 18, 3, 8, 15, 4, 19, 6, 12, 1, 17, 10, 5, 16, 2, 14],
             [13, 18, 3, 8, 15, 4, 19, 6, 12, 1, 17, 10, 5, 16, 2, 14, 7, 20, 11, 9],
             [15, 4, 19, 6, 12, 1, 17, 10, 5, 16, 2, 14, 7, 20, 11, 9, 13, 18, 3, 8],
             [12, 1, 17, 10, 5, 16, 2, 14, 7, 20, 11, 9, 13, 18, 3, 8, 15, 4, 19, 6],
             [5, 16, 2, 14, 7, 20, 11, 9, 13, 18, 3, 8, 15, 4, 19, 6, 12, 1, 17, 10],
             [7, 20, 11, 9, 13, 18, 3, 8, 15, 4, 19, 6, 12, 1, 17, 10, 5, 16, 2, 14]]]
    data = np.load('benchmark/la/la05.npy',allow_pickle=True)
    data = data[0]
    mask = None
    print(data)
    cp_solver = CPSolver()
    sc = cp_solver.solve_blocking_job_shop(data,mask,time_limit=1000,bws=False)
    print("最终makespan:", sc.cal_makespan())
    sc.plotSchedule(data)
    exit(0)

    solver = CPSolver()
    schedule = solver.solve_blocking_job_shop(data)
    schedule.plotSchedule(data)