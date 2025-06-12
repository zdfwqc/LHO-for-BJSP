import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Schedule:
    def __init__(self, num_jobs, num_machines):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.record = []
        self.record_dict = {}
    
    def add_record(self, job_id, operation_id, start_time, end_time):
        # 如果记录已存在，更新结束时间
        if (job_id, operation_id) in self.record_dict:
            old_start_time = self.record_dict[job_id, operation_id][0]
            for i, record in enumerate(self.record):
                if record[0] == job_id and record[1] == operation_id:
                    self.record[i] = (job_id, operation_id, old_start_time, end_time)
                    break
            self.record_dict[job_id, operation_id] = (old_start_time, end_time)
            return
        self.record.append((job_id, operation_id, start_time, end_time))
        self.record_dict[job_id, operation_id] = (start_time, end_time)
        # 作业id，工序id，开始时间，结束时间

    def cal_makespan(self):
        return max([i[3] for i in self.record])

    def cal_utilization(self):
        # 要考虑到所有工序的初始时间不都是0
        totalWorkTime = sum([(i[3]-i[2]) for i in self.record])
        DuringTime = max([i[3] for i in self.record]) - min([i[2] for i in self.record])
        return totalWorkTime / DuringTime

    
    def plotSchedule(self,orginData,savePath = None):
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, self.num_machines+10))
        for j in range(self.num_jobs):
            sequence = orginData[1][j]
            # 考虑到调度结果大概率是部分调度结果，没调度的部分直接跳过
            if (j,0) not in self.record_dict:
                continue
            for i, m in enumerate(sequence):
                start = self.record_dict[j,i][0]
                duration = orginData[0][j][i]
                if duration == 0:
                    continue
                rect = patches.Rectangle(
                    (start, j-0.4),  # 左下角坐标
                    duration,        # 宽度
                    0.8,            # 高度
                    facecolor=colors[m],
                    edgecolor='black',
                    alpha=0.7
                )
                ax.add_patch(rect)
                ax.text(start + duration/2, j, f'M{m}', 
                    ha='center', va='center', color='black', fontweight='bold')
        ax.set_ylim(-0.5, self.num_jobs-0.5)
        ax.set_xlim(0, self.cal_makespan() * 1.1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Job')
        ax.set_title('Job Shop Scheduling Gantt Chart')
        ax.grid(True, linestyle='--', alpha=0.7)
        legend_elements = [patches.Patch(facecolor=colors[m], edgecolor='black', alpha=0.7,
                                    label=f'Machine {m}') for m in range(self.num_machines)]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        if savePath is not None:
            plt.savefig(savePath)
        else:
            plt.show()
            plt.close()


    

        