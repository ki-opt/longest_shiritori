'''
trimmed_dictionaryには{ニントク/ジントク}は含まれない
最長しりとり問題の解法：https://ipsj.ixsq.nii.ac.jp/records/17223
トリビアの種「広辞苑に載っている言葉で最も長くしりとりをすると最後の言葉は○○○」on 2024/3
'''
import time
import pyscipopt
import random as rd
import numpy as np
import pandas as pd
from functools import wraps
from collections import defaultdict

rd.seed(0)

IS_PREPROCESSING_ENABLED = False
NUM_OF_TARGET_WORDS = [1000,5000,10000,30000,50000]

def measure_time(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		start = time.perf_counter()  # 高精度タイマー
		result = func(*args, **kwargs)
		end = time.perf_counter()
		print(f"{func.__name__} took {end - start:.6f} seconds")
		return result
	return wrapper

class JapDictionary:
	DAKUTEN_MAP = str.maketrans(
		"ヴガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ",
		"ウカキクケコサシスセソタチツテトハヒフヘホハヒフヘホ"
	)
	YOUON_MAP = str.maketrans(
		"ァィゥェォヵヶャュョヮ",
		"アイウエオカケヤユヨワ"
	)
	SMALL_KANA_MAP = str.maketrans(
		"ァィゥェォヵヶャュョヮッ",
		"アイウエオカケヤユヨワツ"
	)
	KATAKANA_DICT = {
		'ア': 0, 'イ': 1, 'ウ': 2, 'エ': 3, 'オ': 4,
		'カ': 5, 'キ': 6, 'ク': 7, 'ケ': 8, 'コ': 9,
		'サ': 10, 'シ': 11, 'ス': 12, 'セ': 13, 'ソ': 14,
		'タ': 15, 'チ': 16, 'ツ': 17, 'テ': 18, 'ト': 19,
		'ナ': 20, 'ニ': 21, 'ヌ': 22, 'ネ': 23, 'ノ': 24,
		'ハ': 25, 'ヒ': 26, 'フ': 27, 'ヘ': 28, 'ホ': 29,
		'マ': 30, 'ミ': 31, 'ム': 32, 'メ': 33, 'モ': 34,
		'ヤ': 35, 'ユ': 36, 'ヨ': 37,
		'ラ': 38, 'リ': 39, 'ル': 40, 'レ': 41, 'ロ': 42,
		'ワ': 43, 'ヲ': 44, 'ン': 45
	}
	REVERSE_KATAKANA_DICT = {
		v: k for k, v in KATAKANA_DICT.items()
	}

	def __init__(self):
		'''
		'''
		self.dictionary: pd.DataFrame
		self.f_ij: np.array
		self.f_part_ij: dict
		self.word_list: list

	def read_katarigusa(self):
		'''
		'''
		# ファイル読込
		self.dictionary = pd.read_csv(
			'data/katarigusa.txt', delimiter='\t', header=None, 
			encoding='utf-8', dtype=str
		)
		# カラム名の設定
		self.dictionary.columns = [
			'lemma','reading','POS','conjugation_type','conjugation_form'
		]

	def trim_dictionary(self):
		'''
		'''
		def __extract_normalized_last_char(reading):
			if reading[-1] == 'ー' and len(reading) > 1:
				target = reading[-2]
			else:
				target = reading[-1]

			target = target.translate(self.DAKUTEN_MAP)
			target = target.translate(self.YOUON_MAP)
			target = target.translate(self.SMALL_KANA_MAP)
			return target

		# 名詞の抽出
		self.dictionary = \
			self.dictionary[self.dictionary['POS'].str.contains('名詞',na=False)]
		# 使用する列の抽出
		self.dictionary = self.dictionary[['lemma','reading']]
		# カタカナ・長音の抽出
		self.dictionary = \
			self.dictionary[self.dictionary['reading'].str.match(r'^[ァ-ヶー]+$', na=False)]
		# 最初の文字(読み)の抽出
		self.dictionary['first_char'] = self.dictionary['reading'].str[0].apply(__extract_normalized_last_char)
		# 最初の文字が「ん」「ン」を削除
		self.dictionary = self.dictionary[~self.dictionary['first_char'].isin(['ん','ン'])]
		# 最後の文字(読み)の抽出
		self.dictionary['last_char'] = self.dictionary['reading'].apply(__extract_normalized_last_char)
		# インデックスのリセット
		self.dictionary.reset_index(drop=True,inplace=True)

	def create_graph(self):
		'''
		'''
		i_indices = self.dictionary['first_char'].map(self.KATAKANA_DICT).astype(int)
		j_indices = self.dictionary['last_char'].map(self.KATAKANA_DICT).astype(int)
		f_ij = np.zeros((len(self.KATAKANA_DICT),len(self.KATAKANA_DICT)), dtype=int)
		np.add.at(f_ij,(i_indices.values,j_indices.values),1)
		self.f_ij = f_ij

	def create_tango_dict(self):
		'''
		'''
		i_indices = self.dictionary['first_char'].map(self.KATAKANA_DICT).astype(int)
		j_indices = self.dictionary['last_char'].map(self.KATAKANA_DICT).astype(int)
		word_list = [[[] for _ in range(len(self.KATAKANA_DICT))] for _ in range(len(self.KATAKANA_DICT))]
		for idx in range(len(i_indices)):
			i, j = i_indices[idx], j_indices[idx]
			word = self.dictionary.loc[idx,'lemma']
			word_list[i][j].append(word)
		self.word_list = word_list

	def create_partially_graph(self):
		'''
		'''
		rows = len(self.f_ij)
		cols = len(self.f_ij[0])

		# 1次元に展開
		flat = []
		coords = []

		for i in range(rows):
			for j in range(cols):
					weight = self.f_ij[i][j]
					if weight > 0:
						flat.append(weight)
						coords.append((i, j))
	
		self.f_part_ij = {i:None for i in NUM_OF_TARGET_WORDS}
		for num in self.f_part_ij.keys():
			# 重み付きで num 個抽出（重複あり）
			picked = rd.choices(coords, weights=flat, k=num)

			# 結果用のゼロ配列を作る
			self.f_part_ij[num] = [[0]*cols for _ in range(rows)]

			for i, j in picked:
				self.f_part_ij[num][i][j] += 1
		
		self.f_part_ij.update({sum(sum(row) for row in self.f_ij):self.f_ij.copy()})

	def save_trimmed_dictionary(self):
		'''
		'''
		self.dictionary.to_csv('data/trimmed_katarigusa.csv', index=None)

	def read_trimmed_katarigusa(self):
		'''
		'''
		self.dictionary = pd.read_csv(
			'data/trimmed_katarigusa.csv', encoding='utf-8', dtype=str
		)	

class UnionFind:
	"""
	Union-Find (Disjoint Set Union) データ構造
	グラフの連結成分を効率的に管理・判定
	"""
	def __init__(self, n):
		"""
		初期化: n個の要素（0からn-1）を独立した集合として作成
		
		Args:
			n: 要素数
		"""
		self.n = n
		self.parent = list(range(n))  # 各要素の親（初期は自分自身）
		self.rank = [0] * n           # 木の深さ（ランク）
		self.size = [1] * n           # 各集合のサイズ
		self.num_components = n       # 連結成分の数
	
	def find(self, x):
		"""
		要素xが属する集合の代表元（根）を見つける
		経路圧縮により高速化
		
		Args:
			x: 要素
			
		Returns:
			xが属する集合の代表元
		"""
		if self.parent[x] != x:
			self.parent[x] = self.find(self.parent[x])  # 経路圧縮
		return self.parent[x]
	
	def union(self, x, y):
		"""
		要素xとyが属する集合を統合
		
		Args:
			x, y: 統合する要素
			
		Returns:
			統合が行われた場合True、既に同じ集合の場合False
		"""
		root_x = self.find(x)
		root_y = self.find(y)
		
		if root_x == root_y:
			return False  # 既に同じ集合
		
		# Union by rank: ランクが小さい方を大きい方の下につける
		if self.rank[root_x] < self.rank[root_y]:
			root_x, root_y = root_y, root_x
		
		self.parent[root_y] = root_x
		self.size[root_x] += self.size[root_y]
		
		if self.rank[root_x] == self.rank[root_y]:
			self.rank[root_x] += 1
		
		self.num_components -= 1
		return True
	
	def is_connected(self, x, y) -> bool:
		"""
		要素xとyが同じ連結成分に属するか判定
		
		Args:
			x, y: 判定する要素
			
		Returns:
			同じ連結成分ならTrue
		"""
		return self.find(x) == self.find(y)
	
	def get_size(self, x):
		"""
		要素xが属する連結成分のサイズを取得
		
		Args:
			x: 要素
			
		Returns:
			連結成分のサイズ
		"""
		return self.size[self.find(x)]
	
	def get_num_components(self):
		"""
		全体の連結成分の数を取得
		
		Returns:
			連結成分の数
		"""
		return self.num_components
	
	def is_fully_connected(self):
		"""
		全要素が1つの連結成分になっているか判定
		
		Returns:
			完全連結ならTrue
		"""
		return self.num_components == 1

class Solver:
	def __init__(self, jap_dictionary: JapDictionary):
		'''
		'''
		self.jap_dictionary = jap_dictionary
		self.__V = np.arange(self.jap_dictionary.f_ij.shape[0])	# 頂点集合

	@measure_time
	def solve_by_lp_base_solver(self, given_f_ij=None):
		'''
		'''
		if given_f_ij is None:
			given_f_ij = self.jap_dictionary.f_ij.copy()

		# 定数集合の作成
		self.__s, self.__t = 99, 100							# 開始,終了スーパー頂点
		self.__V_star_l = []
		# モデル定義
		self.__model, self.__x_ij = self.__define_linear_base_problem(given_f_ij)
		# 線形緩和ベース分枝限定法
		z_best = 0
		k = 0
		while (True):
			# 求解
			status = self.__solve()
			# 求解結果の確認
			if status != 'optimal' and status != 'feasible':
				break
			# 目的関数値取得
			z = self.__model.getObjVal()
			# 変数値取得
			x_ij_dict = self.__get_solution()
			# 解の連結性チェック
			is_fully_connected = self.__check_solution_connectivity(x_ij_dict)
			if is_fully_connected:
				if z_best < z:
					z_best = z
					x_best_ij_dict = x_ij_dict
					break
			else:
				if z < z_best:
					break
				z_dash, V_star = self.__get_z_dash_obj_value_and_V_star(x_ij_dict)
				if z_best < z_dash:
					z_best = z_dash
					x_best_ij_dict = x_ij_dict
				# 新しい制約条件を追加した問題の設定
				self.__model, self.__x_ij = self.__define_linear_base_problem(given_f_ij)
				# V*の抽出
				self.__V_star_l.append(V_star)
				# modelに制約の追加
				self.__add_constraint(k+1)
				k += 1

		# dictからnp.arrayへ変更
		first_node, last_node, x_best_ij = self.__transform_x_ij_dict_to_x_ij_np(x_best_ij_dict)

		return first_node, last_node, x_best_ij, z_best - 2

	@measure_time
	def solve_by_construction(self, given_f_ij=None):
		'''
		'''
		if given_f_ij is None:
			given_f_ij = self.jap_dictionary.f_ij.copy()			

		shuffled_list = [i for i in range(len(self.__V)-1)]		# 「ん」を除く
		z_best = 0
		for _ in range(100):
			z = 0
			node = rd.randint(0,len(self.__V)-2)						# 「ん」を除く乱数, randint=[0,10] => (0<=x<=10の乱数)
			first_node = node
			x_ij_list = []
			f_ij = given_f_ij.copy()						# f_ijの設定
			while(True):
				rd.shuffle(shuffled_list)
				is_found = False
				for next_node in shuffled_list:
					if f_ij[node][next_node] > 0:
						z += 1
						x_ij_list.append((node,next_node))
						f_ij[node][next_node] -= 1
						node = next_node
						last_node = node
						is_found = True
						break
				if not is_found:
					break
			# 「ん」に到達するものがあるかチェック
			if f_ij[node][len(self.__V)-1] > 0:
				last_node = len(self.__V)-1
				x_ij_list.append((node,len(self.__V)-1))
				z += 1
			# 最適値の更新
			if z > z_best:
				first_best_node = first_node
				last_best_node = last_node
				x_best_ij_list = x_ij_list.copy()
				z_best = z
								
		# dict形式に変換(線形計画法ベースに合わせるため)
		x_best_ij_dict = defaultdict(int)
		for i in range(0,len(x_best_ij_list),1):
			x_best_ij_dict[x_best_ij_list[i]] += 1
		x_best_ij_dict[(99,first_best_node)] = 1
		x_best_ij_dict[(last_best_node,100)] = 1

		# dictからnp.arrayへ変更
		first_node, last_node, x_best_ij = self.__transform_x_ij_dict_to_x_ij_np(x_best_ij_dict)

		return first_node, last_node, x_best_ij, z_best, x_best_ij_list

	def reconstruction_shiritori(self,
		first_node, 
		last_node,
		x_ij
	):
		'''
		'''
		x_ij = x_ij.copy()
		stack = [first_node]
		trail = []
		while stack:
			u = stack[-1]
			# 未使用の辺を探す
			is_found = False
			for v in range(len(self.__V)):
				if x_ij[u, v] > 0:
					x_ij[u, v] -= 1
					stack.append(v)
					is_found = True
					break  # 1辺だけ進む
			
			if not is_found:
				# 行き詰まったらtrailに追加
				trail.append(stack.pop())

		trail.reverse()

		#print(first_node, last_node)
		#input(trail)
		num = len(self.jap_dictionary.word_list[trail[-2]][trail[-1]])
		return self.jap_dictionary.word_list[trail[-2]][trail[-1]][rd.randint(0,num)-1]
		
	def __define_linear_base_problem(self, given_f_ij:np.array):
		'''
		'''
		### モデルの作成
		model = pyscipopt.Model('longest_shiritori')
		### 変数の作成
		x_ij = {
			(i, j): model.addVar(vtype='I', lb=0, ub=given_f_ij[i][j])
				for i in self.__V for j in self.__V
		}
		for j in self.__V:	# スーパー頂点
			x_ij[(self.__s, j)] = model.addVar(vtype='I', lb=0, ub=1)	# ここ連続変数でも良いかも
			x_ij[(j, self.__t)] = model.addVar(vtype='I', lb=0, ub=1)	# ここ連続変数でも良いかも
		### 目的関数の作成
		model.setObjective(
			pyscipopt.quicksum(x_ij[self.__s, j] for j in self.__V) +
			pyscipopt.quicksum(x_ij[i, j] for i in self.__V for j in self.__V) +
			pyscipopt.quicksum(x_ij[j, self.__t] for i in self.__V),
			'maximize'
		)
		### 制約条件の作成
		# s=>V上にフローを1流す
		model.addCons(pyscipopt.quicksum(x_ij[self.__s, j] for j in self.__V) == 1)
		# s,t,V内での流量保存則
		for i in self.__V:
			model.addCons(
				pyscipopt.quicksum(x_ij[i,j] for j in self.__V) + x_ij[i, self.__t] == 
					pyscipopt.quicksum(x_ij[j,i] for j in self.__V) + x_ij[self.__s, i]
			)
		# V=>t上にフローを1流す
		model.addCons(pyscipopt.quicksum(x_ij[j, self.__t] for j in self.__V) == 1)

		return model, x_ij

	def __solve(self):
		'''
		'''
		### 求解
		self.__model.hideOutput()
		self.__model.optimize()
		return self.__model.getStatus()

	def __get_solution(self) -> dict:
		'''
		'''
		x_ij_dict = {
			key: round(self.__model.getVal(self.__x_ij[key])) 
	 			for key in self.__x_ij.keys()
		}
		return x_ij_dict

	def __check_solution_connectivity(self, x_ij):
		'''
		'''
		s_dash = len(self.__V)
		t_dash = s_dash + 1
		union_find = UnionFind(len(self.__V)+2)
		for i in self.__V:
			for j in self.__V:
				if x_ij[i,j] > 0:
					union_find.union(i,j)
		for j in self.__V:
			if x_ij[self.__s, j] > 0:
				union_find.union(s_dash,j)
			if x_ij[j, self.__t] > 0:
				union_find.union(j,t_dash)
		
		# 連結判定
		# エッジがあるノード（他のノードと同じ集合に属するもの）を抽出
		non_isolated = [v for v in range(len(self.__V)+2) if union_find.get_size(v) > 1]
		
		if len(non_isolated) == 0:
			return True  # 全部孤立ノードなら連結とみなす
		
		root = union_find.find(non_isolated[0])
		return all(union_find.find(v) == root for v in non_isolated)

	def __get_z_dash_obj_value_and_V_star(self, x_ij):
		'''
		'''
		# 最初のひらがな
		for j in self.__V:
			if x_ij[self.__s, j] == 1:
				first_hiragana = int(j)
				break
		# 最後のひらがな
		for j in self.__V:
			if x_ij[j, self.__t] == 1:
				last_hiragana = int(j)
				break

		# z_star(目的関数値), V_star(始点,終点を含む頂点集合)の算出
		z_dash = 2
		idx = 0
		V_star = [first_hiragana, last_hiragana]
		check_node = {first_hiragana, last_hiragana}
		while (idx < len(V_star)):
			i = V_star[idx]
			for j in range(len(self.__V)):
				if j not in check_node and x_ij[i,j] > 0:
					V_star.append(j)
					check_node.add(j)
				z_dash += x_ij[i,j]
			idx += 1
		# スーパーノード(始点,終点)の追加
		V_star.append(self.__s)
		V_star.append(self.__t)
		return z_dash, V_star

	def __add_constraint(self, k):
		'''
		'''
		for l in range(k):
			V_minus_V_star = [i for i in self.__V if i not in self.__V_star_l[l]]
			self.__model.addCons(
				pyscipopt.quicksum(
					self.__x_ij[i,j] if i != self.__t else self.__x_ij[j,i] 
						for i in self.__V_star_l[l] for j in V_minus_V_star
				) >= 1
			)

	def __transform_x_ij_dict_to_x_ij_np(self, x_best_ij_dict):
		'''
		'''
		x_ij = np.zeros_like(self.jap_dictionary.f_ij)
		for key, value in x_best_ij_dict.items():
			if value == 0: 
				continue
			if key[0] == 99:
				first_node = key[1]
			elif key[1] == 100:
				last_node = key[0]
			else:
				x_ij[key[0],key[1]] = value
		return first_node, last_node, x_ij

	def __get_V_star(self, uf:UnionFind):
		'''
		'''
		s_dash = len(self.__V)
		t_dash = s_dash + 1
		V_star = {s_dash, t_dash}
		for i in self.__V:
			if uf.is_connected(s_dash, i):
				V_star.add(i)
			if uf.is_connected(t_dash, i):
				V_star.add(i)
		return list(V_star)

def main():
	jap_dictionary = JapDictionary()
	if IS_PREPROCESSING_ENABLED:
		jap_dictionary.read_katarigusa()
		jap_dictionary.trim_dictionary()
		jap_dictionary.save_trimmed_dictionary()
	else:
		jap_dictionary.read_trimmed_katarigusa()
	jap_dictionary.create_graph()
	jap_dictionary.create_tango_dict()
	jap_dictionary.create_partially_graph()

	for num in jap_dictionary.f_part_ij.keys():
		solver = Solver(jap_dictionary)
		first_node, last_node, x_best_ij, z_best = solver.solve_by_lp_base_solver(jap_dictionary.f_part_ij[num])
		last_word = solver.reconstruction_shiritori(first_node, last_node, x_best_ij)
		print(z_best, last_word)
	
	for num in jap_dictionary.f_part_ij.keys():
		solver = Solver(jap_dictionary)
		first_node, last_node, x_best_ij, z_best, x_best_ij_list = solver.solve_by_construction(jap_dictionary.f_part_ij[num])
		last_word = solver.reconstruction_shiritori(first_node, last_node, x_best_ij)
		print(z_best, last_word)

if __name__ == '__main__':
	main()