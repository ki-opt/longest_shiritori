import pyscipopt


def main():
	# 簡単な線形計画問題を解く関数の実装
	def solve_easy_lp():

		# モデルの作成
		model = pyscipopt.Model("easy_lp")

		# 変数の作成
		x = model.addVar(vtype="C", name = "x", lb=0)
		y = model.addVar(vtype="C", name = "y", lb=0)

		# 目的関数：価値の最大化
		model.setObjective(3*x + 4*y, sense = "maximize")

		# 1つ目の制約条件
		model.addCons(x + 2*y <= 3)

		# 2つ目の制約条件
		model.addCons(4*x + y <= 4)

		# 問題を解く
		model.optimize()

		# 結果の取得
		if model.getStatus() == "optimal":
			x_val = model.getVal(x)
			y_val = model.getVal(y)
			obj_val = model.getObjVal()

			return x_val, y_val, obj_val

	# 計算を実行する
	x_val, y_val, obj_val = solve_easy_lp()

	# 最適解を表示する
	print(f"最適値 : {obj_val}")
	print(f"x : {x_val}")
	print(f"y : {y_val}")

if __name__ == '__main__':
	main()