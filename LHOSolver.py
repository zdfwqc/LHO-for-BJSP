

from network.deepSetNet import DeepSetNet
import torch
import numpy as np
from CPSolver import CPSolver
from netTools import getJobFeature
from schedule import Schedule
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class LHOSolver:

    def __init__(self,numofMachines,k = 5):
        self.PTmask = None
        self.curSelect = None
        self.mask = None
        self.numofJobs = None
        self.numofMachines = numofMachines
        self.CPSolver = CPSolver()
        self.chooseNet = DeepSetNet(input_dim=3*numofMachines)
        self.chooseNet.eval()
        self.k = k

    def reset(self,data):
        assert len(data[0][0]) == self.numofMachines, "初始化网络错误"
        self.numofJobs = len(data[0])
        self.mask = np.zeros(self.numofJobs)
        self.curSelect = []
        self.PTmask = np.zeros((self.numofJobs,self.numofMachines))
    
    def saveNet(self,savePath = 'model/defaultModel'):
        torch.save(self.chooseNet.state_dict(), savePath)

    def loadNet(self,loadPath = 'model/defaultModel'):
        self.chooseNet.load_state_dict(torch.load(loadPath))


    def randomChoose(self,k,randomSeed = 2005):
        # k 为目标选取的数量
        np.random.seed(randomSeed)
        while len(self.curSelect) < k:
            available_jobs = [j for j in range(len(self.mask)) if self.mask[j] == 0 and j not in self.curSelect]
            num_additional = min(k - len(self.curSelect), len(available_jobs))
            if num_additional > 0:
                additional_jobs = np.random.choice(available_jobs, size=num_additional, replace=False)
                self.curSelect.extend(additional_jobs)
                self.mask[additional_jobs] = 1

    def greedyChoose(self,k ,data):
        while len(self.curSelect) < k:
            available_jobs = [j for j in range(len(self.mask)) if self.mask[j] == 0 and j not in self.curSelect]
            if len(available_jobs)  == 0:
                # 没得选了，直接退出
                return 
            available_jobs = [j for j in range(len(self.mask)) if self.mask[j] == 0 and j not in self.curSelect]
            score = []
            for job in available_jobs:
                self.curSelect.append(job)
                jobs_features = getJobFeature(data,self.PTmask,self.curSelect)
                with torch.no_grad():
                    jobs_features = torch.tensor(jobs_features,dtype=torch.float)
                    predicted_utilization = self.chooseNet(jobs_features)
                score.append(predicted_utilization.item())
                self.curSelect.remove(job)
            # 选择得分最高的作业
            #print(score)
            selected_job = available_jobs[np.argmax(score)]
            self.curSelect.append(selected_job)
            self.mask[selected_job] = 1

    def CPGreedyChoose(self,k,data):
        CPSolver = self.CPSolver
        while len(self.curSelect) < k:
            available_jobs = [j for j in range(len(self.mask)) if self.mask[j] == 0 and j not in self.curSelect]
            if len(available_jobs)  == 0:
                # 没得选了，直接退出
                return 
            available_jobs = [j for j in range(len(self.mask)) if self.mask[j] == 0 and j not in self.curSelect]
            score = []
            for job in available_jobs:
                self.curSelect.append(job)
                schedule = CPSolver.solve_blocking_job_shop([data[0][self.curSelect],data[1][self.curSelect]],
                                                            self.PTmask[self.curSelect])
                score.append(schedule.cal_utilization())
                self.curSelect.remove(job)
            # 选择得分最高的作业
            selected_job = available_jobs[np.argmax(score)]
            self.curSelect.append(selected_job)
            self.mask[selected_job] = 1
        
    
    def chooseJobs(self,originData,model = 'net'):
        k = self.k
        if len(self.curSelect) == 0:
            # 第一次选择姑且暂定为随机
            self.randomChoose(k)
            return
        if model == 'random':
            self.randomChoose(k)
        elif model == 'net':
            self.greedyChoose(k,originData)
        elif model == 'greedy':
            self.CPGreedyChoose(k,originData)
        else:
            raise ValueError(f"Invalid model: {model}")
        return 

    def solve(self,data,model = 'net',clipModel = 'earliest' ,clipCount = 12):
        cpSolver = CPSolver()
        numofJobs = len(data[0])
        numofMachines = len(data[0][0])
        assert numofMachines == self.numofMachines, "网络宽度不匹配错误"
        self.reset(data)

        # 考虑一下其他的截断方式，比如保留一定次数的截断
        totalSchedule = Schedule(numofJobs,numofMachines)
        sovleCount = 0
        baseTime = 0

        while sovleCount < numofJobs:

            self.chooseJobs(originData=data,model = model)
            schedule = cpSolver.solve_blocking_job_shop([data[0][self.curSelect],data[1][self.curSelect]],
                                                        self.PTmask[self.curSelect])
            for record in schedule.record:
                totalSchedule.add_record(int(self.curSelect[record[0]]),record[1],baseTime+record[2],baseTime+record[3])

            job_completion_times = {}
            for job, machine, start_time, end_time in schedule.record:
                job = self.curSelect[job]
                # 转成真实的job
                if job not in job_completion_times or end_time > job_completion_times[job]:
                    job_completion_times[job] = end_time

            earliest_complete_job = min(job_completion_times.items(), key=lambda x: x[1])
            earliest_job_id = earliest_complete_job[0]
            earliest_completion_time = earliest_complete_job[1]

            # 更新记录
            for tem_job, machine, start_time, end_time in schedule.record:
                job = self.curSelect[tem_job]
                if (end_time - start_time) == 0:
                    continue
                if end_time <= earliest_completion_time:
                    self.PTmask[job][machine] = -1
                elif end_time > earliest_completion_time > start_time:
                    self.PTmask[job][machine] = data[0][job][machine] - earliest_completion_time + start_time

            baseTime = baseTime + earliest_completion_time

            # 更新mask,标记完成的工件
            # totalSchedule.plotSchedule(data)
            sovleCount += 1
            self.curSelect.remove(earliest_job_id)
            if len(self.curSelect) <= 1:
                break

        makespan = totalSchedule.cal_makespan()
        # 计算剩余作业的总加工时间
        remaining_time = 0
        for job_id in self.curSelect:
            for machine in range(numofMachines):
                if self.PTmask[job_id][machine] > 0:  # 如果是负值
                    remaining_time += self.PTmask[job_id][machine]
                elif self.PTmask[job_id][machine] == 0:
                    remaining_time += data[0][job_id][machine]
        makespan = makespan + remaining_time
        return makespan




if __name__ == "__main__":
    data = np.load('benchmark/la/la11.npy',allow_pickle=True)
    data = data[0]
    solver = LHOSolver(numofMachines=5,k = 5)
    solver.reset(data)
    solver.loadNet('model/dsh75_NORM.pth')
    print(solver.solve(data,model='net'))

    print(solver.solve(data, model='greedy'))