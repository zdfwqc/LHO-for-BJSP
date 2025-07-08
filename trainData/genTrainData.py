import numpy as np
from CPSolver import CPSolver
from uniform_instance_gen import uni_instance_gen
from memory import Memory
import time
from tqdm import tqdm
from netTools import getJobFeature
from  schedule import Schedule
from params import configs


def generate_memory_data(num_instances=1000, n_jobs=5, m_machines=5, memory_size=100000,time_limit=100):
    memory = Memory(num_machines=m_machines, capacity=memory_size)
    print(f"开始生成{num_instances}个训练实例...")
    for i in tqdm(range(num_instances)):
        data = uni_instance_gen(n_jobs, m_machines, 1, 100)
        ptMask = np.zeros((n_jobs, m_machines))
        # 对每一行都以0.05的概率将数据清零
        for row in range(n_jobs):
            if np.random.random() < 0.03:
                # 将该行的加工时间和机器序列都设为0
                data[0][row] = [0] * m_machines
                data[1][row] = [0] * m_machines
                for j in range(m_machines):
                    ptMask[row, j] = -1
        # 对每一行都以0.1的概率将前几个工序清零
        OCMachine = set()
        for row in range(n_jobs):
            if np.random.random() < 0.2:
                # 随机选择要清零的工序数量(1到m_machines之间)
                zero_ops = np.random.randint(1, m_machines-1)
                # 不能预占相同的机器，不能清空后再去清
                if data[0][row][zero_ops] == 0:
                    continue
                if data[1][row][zero_ops] not in OCMachine:
                    OCMachine.add(data[1][row][zero_ops])
                else:
                    continue
                # 将该行的前zero_ops个工序的加工时间和机器序列都设为0
                data[0][row][:zero_ops] = [0] * zero_ops
                data[1][row][:zero_ops] = [0] * zero_ops
                for i in range(zero_ops):
                    ptMask[row, i] = -1
                ptMask[row, zero_ops ] = np.random.randint(1, data[0][row][zero_ops]+1)

        # 使用CP求解器求解
        cpSolver = CPSolver()
        schedule = cpSolver.solve_blocking_job_shop(data,ptMask,time_limit=100)
        jobs_features = getJobFeature(data, ptMask,range(n_jobs))
        memory.push(jobs_features, schedule.cal_utilization())

    print("数据生成完成！")

    return memory


if __name__ == "__main__":
    # 生成数据并存储到memory
    memory = generate_memory_data(
        num_instances=configs.gen_instance_num,  # 生成1000个实例
        n_jobs=configs.gen_job_num,  # 10个工件
        m_machines=configs.gen_machine_num,  # 5台机器
        memory_size=200000 , # memory大小
        time_limit=configs.gen_time_limit
    )
    # memory.save('TestData_7_5')
    memory.save('TrainData' +str(configs.gen_instance_num) + '_' + str(configs.gen_job_num) + '_' + str(configs.gen_machine_num) + '_' + str(configs.gen_time_limit))