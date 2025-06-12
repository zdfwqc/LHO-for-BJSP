def getJobFeature(orginData,PTmask,curSelected):
    numofMachines = len(orginData[0][0])
    numofJobs = len(curSelected)
    jobs_features = [[] for i in range(numofJobs)]
    for i in range(numofJobs):
        for j in range(numofMachines):
            if PTmask[curSelected[i]][j] == -1:
                # 已经完成的工序
                # 机器设为0是一个特殊值 正常机器不会为0
                jobs_features[i].append(0)
                jobs_features[i].append(0)
                jobs_features[i].append(0)
            elif PTmask[curSelected[i]][j] >0:
                # 未完全完成的工序 强制要求开头开始
                jobs_features[i].append(PTmask[curSelected[i]][j])
                jobs_features[i].append(orginData[1][curSelected[i]][j])
                jobs_features[i].append(1)
            else:
                # 正常的工序
                jobs_features[i].append(orginData[0][curSelected[i]][j])
                jobs_features[i].append(orginData[1][curSelected[i]][j])
                jobs_features[i].append(0)
    return jobs_features