import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


class BJSPGraph:
    def __init__(self, time_mat, machine_mat, defaultWeight=0.001):
        self.time_mat = time_mat
        self.machine_mat = machine_mat
        self.defaultWeight = defaultWeight
        self.n_jobs = len(time_mat)
        self.n_machines = len(time_mat[0])
        self.graph = nx.DiGraph()
        self.alternative_edges = [] 
        self.blocks =[]
        self._build_graph()
        
    class CriticalBlock:
        def __init__(self, type_id, op1, op2, first_edge_type=None):
            self.type_id = type_id  # 1或2
            self.op1 = op1
            self.op2 = op2
            self.first_edge_type = first_edge_type  # 对于type_id=1，记录连接op1的边类型
        def __repr__(self):
            if self.type_id == 1:
                return f"Block(type={self.type_id}, {self.op1}, {self.op2}, first_edge={self.first_edge_type})"
            else:
                return f"Block(type={self.type_id}, {self.op1}, {self.op2})"
    def _build_graph(self):
        for j in range(self.n_jobs):
            for o in range(self.n_machines):
                node = f"J{j}_O{o}"
                self.graph.add_node(node, job=j, oper=o, machine=self.machine_mat[j][o], proc_time=self.time_mat[j][o], dummy=False)
        # 添加虚拟节点（编号为last_o+1，machine=0）
        for j in range(self.n_jobs):
            last_o = self.n_machines - 1
            dummy = f"J{j}_O{last_o+1}"
            self.graph.add_node(dummy, job=j, oper=last_o+1, machine=0, proc_time=0, dummy=True)
        # 添加工艺边（同一作业内的顺序约束，包括虚拟节点）
        for j in range(self.n_jobs):
            for o in range(self.n_machines):
                u = f"J{j}_O{o}"
                v = f"J{j}_O{o+1}"
                if self.graph.has_node(v):
                    self.graph.add_edge(u, v, type='tech', weight=self.time_mat[j][o])
        # 添加替代边（仅对machine>=1的工序）
        self.alternative_edges = []
        for m in range(1, self.n_machines+1):  # 机器编号从1开始
            ops_on_m = []
            for j in range(self.n_jobs):
                for o in range(self.n_machines):
                    if self.machine_mat[j][o] == m:
                        ops_on_m.append((j, o))
            for i in range(len(ops_on_m)):
                for k in range(i+1, len(ops_on_m)):
                    j1, o1 = ops_on_m[i]
                    j2, o2 = ops_on_m[k]
                    a_succ = f"J{j1}_O{o1+1}" if self.graph.has_node(f"J{j1}_O{o1+1}") else None
                    b_succ = f"J{j2}_O{o2+1}" if self.graph.has_node(f"J{j2}_O{o2+1}") else None
                    if a_succ and b_succ:
                        edge_pair = {
                            'A_succ_to_B': (a_succ, f"J{j2}_O{o2}"),
                            'B_succ_to_A': (b_succ, f"J{j1}_O{o1}"),
                            'in_graph': None  # None表示都未加入，'A'表示A_succ_to_B已加入，'B'表示B_succ_to_A已加入
                        }
                        self.alternative_edges.append(edge_pair)
        # 默认不加入任何替代边

    def update_alternative_edge(self, edge_pair, which):
        # which: 'A' 或 'B'，表示加入哪一条边，另一条不加
        # 获取边对中的两条边
        a = edge_pair['A_succ_to_B']
        b = edge_pair['B_succ_to_A']
        
        # 加入指定边
        if which == 'A':
            if not self.graph.has_edge(*a):
                self.graph.add_edge(*a, type='alternative', weight=self.defaultWeight)
            edge_pair['in_graph'] = 'A'
        elif which == 'B':
            if not self.graph.has_edge(*b):
                self.graph.add_edge(*b, type='alternative', weight=self.defaultWeight)
            edge_pair['in_graph'] = 'B'
        else:
            edge_pair['in_graph'] = None

    def visualize(self, save_path=None, figsize=(12,8)):
        # N*M矩阵布局
        pos = {}
        for node in self.graph.nodes():
            node_attr = self.graph.nodes[node]
            j = node_attr.get('job', 0)
            o = node_attr.get('oper', 0)
            pos[node] = (o, -j)  # x=工序编号, y=作业编号（反向让第0行在上）
        # 为每个节点分配颜色：同一机器同色
        machine_colors = plt.cm.get_cmap('tab10', self.n_machines+1)
        node_colors = []
        for node in self.graph.nodes():
            machine = self.graph.nodes[node].get('machine', 0)
            node_colors.append(machine_colors(machine))
        plt.figure(figsize=figsize)
        # 画节点
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=700)
        # 画工艺边
        tech_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d['type'] == 'tech']
        nx.draw_networkx_edges(
            self.graph, pos, edgelist=tech_edges, edge_color='tab:blue', width=2, arrows=True,
            label='job', arrowsize=30, arrowstyle='-|>'
        )
        # 画替代边
        alternative_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d['type'] == 'alternative']
        nx.draw_networkx_edges(
            self.graph, pos, edgelist=alternative_edges, edge_color='tab:green', style='dashed', width=1, arrows=True,
            label='alternative', arrowsize=30, arrowstyle='-|>'
        )
        # 画标签
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_color='black')
        # 图例
        for m in range(self.n_machines):
            plt.scatter([], [], color=machine_colors(m), label=f'machine{m}')
        plt.plot([], [], color='tab:blue', linewidth=2, label='job')
        plt.plot([], [], color='tab:green', linestyle='dashed', linewidth=1, label='alternative')
        plt.legend()
        plt.title('BJSP')
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()

    def generate_random_initial_solution(self, seed=None):
        """
        随机生成一个工件顺序，并据此为所有替代边做出选择。
        选中的边会直接加入图中。
        返回：工件顺序（列表）
        """
        if seed is not None:
            random.seed(seed)
        job_order = list(range(self.n_jobs))
        random.shuffle(job_order)
        # job_order_map[job_id] = 在排列中的位置
        job_order_map = {job: idx for idx, job in enumerate(job_order)}
        # 先移除所有替代边
        for edge_pair in self.alternative_edges:
            self.update_alternative_edge(edge_pair, None)
        # 再按顺序加入
        for edge_pair in self.alternative_edges:
            a_succ, b = edge_pair['A_succ_to_B']
            b_succ, a = edge_pair['B_succ_to_A']
            a_job = int(a.split('_')[0][1:])
            b_job = int(b.split('_')[0][1:])
            if job_order_map[a_job] < job_order_map[b_job]:
                self.update_alternative_edge(edge_pair, 'A')
            else:
                self.update_alternative_edge(edge_pair, 'B')
        return job_order

    def has_positive_cycle(self):
        """
        检测图中是否存在正环（权值和大于0的有向环）。
        返回True表示有正环，False表示无正环。
        """
        # 使用Bellman-Ford思想：对每个节点做一次松弛
        nodes = list(self.graph.nodes())
        edges = list(self.graph.edges(data=True))
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}
        # 初始化所有点的距离为0
        dist = [0.0] * n
        # 进行n次松弛
        for _ in range(n):
            updated = False
            for u, v, d in edges:
                w = d.get('weight', 0)
                if dist[node_idx[v]] < dist[node_idx[u]] + w:
                    dist[node_idx[v]] = dist[node_idx[u]] + w
                    updated = True
            if not updated:
                break
        # 再做一次松弛，如果还能更新，说明有正环
        for u, v, d in edges:
            w = d.get('weight', 0)
            if dist[node_idx[v]] < dist[node_idx[u]] + w:
                return True
        return False

    def topological_sort(self):
        """
        返回当前图的一个拓扑序列（节点列表）。
        如果有环则抛出networkx.exception.NetworkXUnfeasible异常。
        """
        return list(nx.topological_sort(self.graph))

    def longest_path_with_edge_type(self):
        """
        找到当前图的最长路径，并打印路径及每对相邻节点的连接边类型（工艺边/替代边）。
        返回：(path, edge_types)
        """
        import networkx as nx
        # 只考虑有向无环图
        if not nx.is_directed_acyclic_graph(self.graph):
            print("图中有环，无法计算最长路径")
            return None, None
        # 计算最长路径
        path = nx.dag_longest_path(self.graph, weight='weight')
        edge_types = []
        path_length = 0
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            edge_type = self.graph[u][v].get('type', 'unknown')
            edge_types.append(edge_type)
            path_length += self.graph[u][v].get('weight', 0)
        # 打印
        print("最长路径:")
        for i in range(len(path)):
            print(path[i], end='')
            if i < len(edge_types):
                if edge_types[i] == 'tech':
                    print(" --[工艺边]--> ", end='')
                elif edge_types[i] == 'alternative':
                    print(" --[替代边]--> ", end='')
                else:
                    print(f" --[{edge_types[i]}]--> ", end='')
        print()
        print(f"最长路径长度: {path_length}")
        return path, edge_types


    def find_critical_blocks(self):
        """
        查找所有1类和2类关键块，返回列表。
        1类关键块：最长路径上连续三节点a-b-c，b是a的工艺后继（同作业且工序号+1），ac同机器。
        2类关键块：剩余未覆盖的替代边，若a->b在最长路径，则a的前继和b构成2类关键块。
        返回的关键块按op1在拓扑序中的顺序排序。
        """
        path, _ = self.longest_path_with_edge_type()
        if not path:
            return []
        blocks = []
        used_alternative = set()
        for i in range(len(path)-2):
            a, b, c = path[i], path[i+1], path[i+2]
            a_job = self.graph.nodes[a].get('job', -1)
            b_job = self.graph.nodes[b].get('job', -2)
            a_oper = self.graph.nodes[a].get('oper', -1)
            b_oper = self.graph.nodes[b].get('oper', -2)
            if a_job == b_job and b_oper == a_oper + 1:
                if self.graph.nodes[a].get('machine', -1) == self.graph.nodes[c].get('machine', -2) and self.graph.nodes[a].get('machine', -1) > 0:
                    # 找到连接a的边类型
                    first_edge_type = None
                    if i > 0:
                        prev_node = path[i-1]
                        if self.graph.has_edge(prev_node, a):
                            first_edge_type = self.graph[prev_node][a].get('type', 'unknown')
                    blocks.append(self.CriticalBlock(1, a, c, first_edge_type))
                    used_alternative.add((b, c))
        path_set = set(path)
        for edge_pair in self.alternative_edges:
            for which, (src, dst) in [('A', edge_pair.get('A_succ_to_B')), ('B', edge_pair.get('B_succ_to_A'))]:
                if (src, dst) in used_alternative:
                    continue
                if src in path_set and dst in path_set:
                    idx_src = path.index(src)
                    idx_dst = path.index(dst)
                    if idx_dst == idx_src + 1:
                        if idx_src > 0:
                            pred = path[idx_src-1]
                            blocks.append(self.CriticalBlock(2, pred, dst))
        # 按op1的拓扑序排序
        topo = {node: i for i, node in enumerate(self.topological_sort())}
        blocks.sort(key=lambda block: topo.get(block.op1, 1<<30))
        self.blocks = blocks
        return blocks
    
    def forward(self, block):
        # 这里简单假设block长度为2
        self.change(block.op1,block.op2)
        

    def backward(self,block):
        assert(block.type_id == 1)
        if block.first_edge_type =='tech':
            self.change(block.op2,block.op1)
        else :
            # 好难写啊
            # 获取拓扑序
            print('2 test')
            topo_order = self.topological_sort()
            topo_idx = {node: i for i, node in enumerate(topo_order)}
            
            # 获取block.op1的机器号
            op1_machine = self.graph.nodes[block.op1].get('machine', -1)
            op1_topo_idx = topo_idx[block.op1]
            
            # 找到拓扑序上block.op1用同样机器的上一个节点
            prev_node = None
            min_topo_idx = float('inf')
            for node in self.graph.nodes():
                if self.graph.nodes[node].get('machine', -1) == op1_machine:
                    node_topo_idx = topo_idx[node]
                    if node_topo_idx > op1_topo_idx and node_topo_idx < min_topo_idx:
                        prev_node = node
                        min_topo_idx = node_topo_idx
            
            if prev_node:
                self.change(block.op2, prev_node)
            pass


    def change(self, u, l):
        """
        切换两个节点u和l的邻域操作
        找到指向u的替代边ab（多条选择a拓扑序最低的）
        找到u的后继节点指向别人的替代边cd（多条选择d拓扑序最高的）
        找到l节点的后继指向别人的替代边ef（多条选择f拓扑序最高的）
        删除这些边，然后加入ad, eb, cf三条替代边
        """
        # 获取拓扑序
        topo_order = self.topological_sort()
        
        # 找到u的后继节点（工序号+1的节点）
        u_job = self.graph.nodes[u].get('job', -1)
        u_oper = self.graph.nodes[u].get('oper', -1)
        u_succ = f"J{u_job}_O{u_oper+1}"
        
        # 找到l的后继节点（工序号+1的节点）
        l_job = self.graph.nodes[l].get('job', -1)
        l_oper = self.graph.nodes[l].get('oper', -1)
        l_succ = f"J{l_job}_O{l_oper+1}"
        
        # 1. 找到节点集合A，集合里面的节点到u在图中存在替代边
        nodes_to_u = set()
        for edge_pair in self.alternative_edges:
            for which, (src, dst) in [('A', edge_pair.get('A_succ_to_B')), ('B', edge_pair.get('B_succ_to_A'))]:
                if dst == u and self.graph.has_edge(src, dst):
                    nodes_to_u.add(src)
        
        
        # 2. 找到u的后继节点指向别人的替代边cd（多条选择d拓扑序最高的）
        nodes_from_u_succ = set()
        if self.graph.has_node(u_succ):
            for edge_pair in self.alternative_edges:
                for which, (src, dst) in [('A', edge_pair.get('A_succ_to_B')), ('B', edge_pair.get('B_succ_to_A'))]:
                    if src == u_succ and self.graph.has_edge(src, dst):
                        nodes_from_u_succ.add(dst)

        
        # 3. 找到l节点的后继指向别人的替代边ef（多条选择f拓扑序最高的）
        nodes_from_l_succ = []
        if self.graph.has_node(l_succ):
            for edge_pair in self.alternative_edges:
                for which, (src, dst) in [('A', edge_pair.get('A_succ_to_B')), ('B', edge_pair.get('B_succ_to_A'))]:
                    if src == l_succ and self.graph.has_edge(src, dst):
                        nodes_from_l_succ.append(dst)
        
    
        # 删除之前找到的替代边
        # 删除nodes_to_u中指向u的替代边
        for src in nodes_to_u:
            if self.graph.has_edge(src, u):
                self.graph.remove_edge(src, u)
        # 删除u的后继节点指向别人的替代边
        if self.graph.has_node(u_succ):
            for dst in nodes_from_u_succ:
                if self.graph.has_edge(u_succ, dst):
                    self.graph.remove_edge(u_succ, dst)
        # 删除l的后继节点指向别人的替代边
        if self.graph.has_node(l_succ):
            for dst in nodes_from_l_succ:
                if self.graph.has_edge(l_succ, dst):
                    self.graph.remove_edge(l_succ, dst)
        
        # 添加nodes_to_u内部节点到nodes_from_u_succ节点的边
        for src in nodes_to_u:
            for dst in nodes_from_u_succ:
                if not self.graph.has_edge(src, dst):
                    self.graph.add_edge(src, dst, type='alternative', weight=self.defaultWeight)
        # 添加l节点后继到u的边
        if self.graph.has_node(l_succ):
            if not self.graph.has_edge(l_succ, u):
                self.graph.add_edge(l_succ, u, type='alternative', weight=self.defaultWeight)
        # 添加u后继到nodes_from_l_succ节点的边
        if self.graph.has_node(u_succ):
            for dst in nodes_from_l_succ:
                if not self.graph.has_edge(u_succ, dst):
                    self.graph.add_edge(u_succ, dst, type='alternative', weight=self.defaultWeight)

    def JIFR1():
        
        pass
# 示例用法
if __name__ == '__main__':
    # 示例数据（5作业5工序）
    time_mat = np.array([
        [45, 78, 23],
        [12, 89, 34], 
        [71, 39, 95],
    ])
    machine_mat = np.array([
        [1, 2, 3],
        [2, 1, 3],
        [3, 1, 2],
    ])
    g = BJSPGraph(time_mat, machine_mat)
    g.generate_random_initial_solution()
    g.visualize()
    print(g.has_positive_cycle())
    print(g.topological_sort())
    g.longest_path_with_edge_type()
    blocks = g.find_critical_blocks()
    print(g.find_critical_blocks())
    g.forward(blocks[0])
    g.visualize()
    # 当前版本的邻域不是很准确 不过不要紧
