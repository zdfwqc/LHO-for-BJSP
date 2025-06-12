import argparse
parser = argparse.ArgumentParser(description='阻塞作业车间调度问题求解器')
    
    # 添加参数
parser.add_argument('--jobs', type=int, default=10,
                      help='作业数量 (默认: 10)')
parser.add_argument('--machines', type=int, default=5,
                      help='机器数量 (默认: 5)')
parser.add_argument('--time-limit', type=int, default=300,
                      help='求解时间限制（秒）(默认: 300)')
parser.add_argument('--dataPath', type=str,default='/la',
                      help='输入文件路径，包含加工时间和机器顺序数据')
parser.add_argument('--max_updates', type=int,default=10000)

parser.add_argument('--epochs', type=int, default=1000)


configs = parser.parse_args()