import numpy as np

class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G  # 存储着图的边有无权重/方向 无向无权
		self.is_directed = is_directed

		self.p = p#两个超参
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		appearnum2 = np.zeros((3000))
		appearnum2[start_node]=1
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges
		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]  # 这样无论再怎么加都是考虑后一个节点的下一个节点比如a 下一个是b 再下一个就考虑b
			cur_nbrs = sorted(G.neighbors(cur))  # 考虑当前邻居节点的集合
			if len(cur_nbrs) > 0:   # 如果邻居节点个数大于零
				# print(len(cur_nbrs),cur,'length')
				if len(walk) == 1:   # 如果是第一步则只考虑节点概率进行游走
					a=cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
					walk.append(a)  # 通过alias_draw 得到节点的index 数是随机的
					appearnum2[a]+=1
				else:                 # 如果是第二步及以后则需要考虑上一步的节点
					prev = walk[-2]   # 上一步的节点则是walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]   # 通过alias_draw 得到节点的index 然后通过邻居节点的集合找到下标对应的节点
					walk.append(next)
					appearnum2[next]+=1

					# print(walk,appearnum2,start_node)
			else:
				break

		return walk,appearnum2

	def simulate_walks(self, num_walks, walk_length):#10 80
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		appearnum = []      # appearnum是list，里面append的appearnum2是矩阵
		nodes = list(G.nodes())
		abcList = []
		for i in range(2000):#973 2000
			if i not in nodes:
				abcList.append((i,0))
			# return returnMat

		print('abclist',abcList)
		print(len(abcList),len(nodes))




		nodes.sort(reverse=False)
		# print('nodes序列',nodes,len(nodes))
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(str(walk_iter+1), '/', str(num_walks))

			for node in nodes:
				walk,appearnum2=self.node2vec_walk(walk_length=walk_length, start_node=node)
				appearnum.append(appearnum2)
				walks.append(walk)
		return walks,appearnum,nodes

	def get_alias_edge(self, src, dst):  # 上一个节点 当前节点
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:           # 如果邻居节点是上一个节点的话
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)  # 未标准化概率记为该条边的权值除以p
			elif G.has_edge(dst_nbr, src):  # 如果邻居节点是和上一个节点有边的化
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])                      # 未标准化的概率记为这条边的权值
			else:                                                                         # 如果邻居节点和上一个节点没有边的话
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)                    # 未标准话的概率记为这条边的权值除以q
		norm_const = sum(unnormalized_probs)                                              # 有邻居未标准的概率求和
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]   # 将其除以总和得到标准化后的概率

		return alias_setup(normalized_probs)                                              # 对标准化的结果进行采样

	def preprocess_transition_probs(self):                                                     # 预备转移概率矩阵
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G                                                                              # 事先生成的图
		is_directed = self.is_directed                                                          # 无向

		alias_nodes = {}
		for node in G.nodes():                                                                  # G 所有的节点
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]  # 目标节点所有邻居的权重
			norm_const = sum(unnormalized_probs)                                                # 目标节点所有邻居的权重之和
			# print(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]     # 对每个权重除以总和形成转移概率矩阵即标准化
		    # print(normalized_probs,'nomalized')
			alias_nodes[node] = alias_setup(normalized_probs)                                   # 进行采样 得到aliasnodes[node]
		    # print(alias_nodes)

		alias_edges = {}                                                                        # 定义采样边 字典
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:                                                                                   # 无向
			for edge in G.edges():
				# print(edge)# 对G图里的每一条边（1.txt，32）
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])                       # edge 0代表上一条边source edge，1代表当前边destination edge
				# print(alias_edges)
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])     #加一条反方向的边（32，1.txt）
		self.alias_nodes = alias_nodes
		# print(alias_nodes,'hhhhhh')
		self.alias_edges = alias_edges
		# print(alias_edges)

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]
