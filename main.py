'''
trimmed_dictionaryには{ニントク/ジントク}は含まれない
最長しりとり問題の解法：https://ipsj.ixsq.nii.ac.jp/records/17223
トリビアの種「広辞苑に載っている言葉で最も長くしりとりをすると最後の言葉は○○○」on 2024/3
'''

import pyscipopt
import numpy as np
import pandas as pd

IS_PREPROCESSING_ENABLED = True

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
	
	def is_connected(self, x, y):
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

	def solve_by_lp_base_solver(self):
		'''
		'''
		# 定数集合の作成
		self.__f_ij = self.jap_dictionary.f_ij.copy()
		self.__V = np.arange(self.__f_ij.shape[0])
		self.__s, self.__t = int(0), int(0)
		# モデル定義
		self.__model, self.__x_ij, self.__x_sj, self.__x_jt = \
			self.__define_linear_base_problem()
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
			x_ij, x_sj, x_jt = self.__get_solution()
			# 解の連結性チェック 
			if self.__check_solution_connectivity(x_ij, x_sj, x_jt):
				if z_best < z:
					z_best = z
					x_best_ij, x_best_sj, x_best_jt = x_ij, x_sj, x_jt
					break
			else:
				if z < z_best:
					break
				z_dash = self.__get_z_dash_obj_value(x_ij, x_sj, x_jt)
				if z_best < z_dash:
					z_best = z_dash
					x_best_ij, x_best_sj, x_best_jt = x_ij, x_sj, x_jt
				# 新しい制約条件を追加した問題の設定
				self.__model, self.__x_ij, self.__x_sj, self.__x_jt = \
					self.__define_linear_base_problem()
				
				###########self.__dfklj;adfklj;sadfjkl => 制約追加まだ
				k += 1

		# 得られた解からしりとりの構成
		x_best_ij, x_best_sj, x_best_jt

	def __define_linear_base_problem(self):
		'''
		'''
		### モデルの作成
		model = pyscipopt.Model('longest_shiritori')
		### 変数の作成
		x_ij = {
			(i, j): model.addVar(vtype='I', lb=0, ub=self.__f_ij[i,j])
				for i in self.__V for j in self.__V
		}
		x_sj = {
			(self.__s, j): model.addVar(vtype='C', lb=0, ub=1) for j in self.__V
		}
		x_jt = {
			(j, self.__t): model.addVar(vtype='C', lb=0, ub=1) for j in self.__V
		}
		### 目的関数の作成
		model.setObjective(
			pyscipopt.quicksum(x_sj[self.__s,j] for j in self.__V) + 
			pyscipopt.quicksum(x_ij[i,j] for i in self.__V for j in self.__V) + 
			pyscipopt.quicksum(x_jt[j,self.__t] for j in self.__V),
			'maximize'
		)
		### 制約条件の作成
		# s=>V上にフローを1流す
		model.addCons(pyscipopt.quicksum(x_sj[self.__s,j] for j in self.__V) == 1)
		# s,t,V内での流量保存則
		for i in self.__V:
			model.addCons(
				pyscipopt.quicksum(x_ij[i,j] for j in self.__V) + x_jt[i,self.__t] == 
					pyscipopt.quicksum(x_ij[j,i] for j in self.__V) + x_sj[self.__s,i]
			)
		# V=>t上にフローを1流す
		model.addCons(pyscipopt.quicksum(x_jt[j,self.__t] for j in self.__V) == 1)

		return model, x_ij, x_sj, x_jt

	def __solve(self):
		'''
		'''
		### 求解
		self.__model.optimize()
		return self.__model.getStatus()

	def __get_solution(self) -> pd.DataFrame:
		'''
		'''
		x_ij_np = np.array([
			[round(self.__model.getVal(self.__x_ij[i,j])) for j in self.__V] 
				for i in self.__V
		])
		x_sj_np = np.array([
			[round(self.__model.getVal(self.__x_sj[self.__s,j])) for j in self.__V]
		])
		x_jt_np = np.array([
			[round(self.__model.getVal(self.__x_jt[j,self.__t]))] for j in self.__V
		])
		return x_ij_np, x_sj_np, x_jt_np

	def __check_solution_connectivity(self, x_ij, x_sj, x_jt):
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
			if x_sj[self.__s,j] > 0:
				union_find.union(s_dash,j)
			if x_jt[j,self.__t] > 0:
				union_find.union(j,t_dash)
		
		# 連結判定
		if union_find.is_fully_connected():
			return True
		return False

	def __get_z_dash_obj_value(self, x_ij, x_sj, x_jt):
		'''
		'''
		# 最初のひらがな
		first_hiragana = -1
		for j in self.__V:
			if x_sj[self.__s,j] == 1:
				first_hiragana = int(j)
				break
		# 最後のひらがな
		last_hiragana = -1 
		for j in self.__V:
			if x_jt[j,self.__s] == 1:
				last_hiragana = int(j)
				break
		# z_star, V_starの算出
		z_dash = 2
		idx_V_star = 0
		V_star = [first_hiragana, last_hiragana]
		check_node = {first_hiragana, last_hiragana}
		while (idx_V_star < len(V_star)):
			i = V_star[idx_V_star]
			for j in range(len(self.__V)):
				if j not in check_node and x_ij[i,j] > 0:
					V_star.append(j)
					check_node.add(j)
				z_dash += x_ij[i,j]
			idx_V_star += 1
		input(z_dash)
		input(V_star)
		return z_dash


def main():
	jap_dictionary = JapDictionary()
	if IS_PREPROCESSING_ENABLED:
		jap_dictionary.read_katarigusa()
		jap_dictionary.trim_dictionary()
		jap_dictionary.save_trimmed_dictionary()
	else:
		jap_dictionary.read_trimmed_katarigusa()
	jap_dictionary.create_graph()

	solver = Solver(jap_dictionary)
	solver.solve_by_lp_base_solver()

if __name__ == '__main__':
	main()