'''
trimmed_dictionaryには{ニントク/ジントク}は含まれない
'''

import pyscipopt
import pandas as pd

IS_PREPROCESSING_ENABLED = False

class JapDictionary:
	DAKUTEN_MAP = str.maketrans(
		"ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ",
		"カキクケコサシスセソタチツテトハヒフヘホハヒフヘホ"
	)

	def __init__(self):
		'''
		'''
		self.dictionary: pd.DataFrame
		

	def readKatarigusa(self):
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

	def trimDictionary(self):
		'''
		'''
		def __extract_normalized_last_char(reading):
			if reading[-1] == 'ー' and len(reading) > 1:
				target = reading[-2]
			else:
				target = reading[-1]

			target = target.translate(self.DAKUTEN_MAP)
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
		self.dictionary['first_char'] = self.dictionary['reading'].str[0]
		# 最後の文字(読み)の抽出
		self.dictionary['last_char'] = self.dictionary['reading'].apply(__extract_normalized_last_char)
		# インデックスのリセット
		self.dictionary.reset_index(drop=True,inplace=True)

	def saveTrimmedDictionary(self):
		'''
		'''
		self.dictionary.to_csv('data/trimmed_katarigusa.csv', index=None)

	def readTrimmedKatarigusa(self):
		'''
		'''
		self.dictionary = pd.read_csv(
			'data/trimmed_katarigusa.csv', encoding='utf-8', dtype=str
		)	


def main():
	jap_dictionary = JapDictionary()
	if IS_PREPROCESSING_ENABLED:
		jap_dictionary.readKatarigusa()
		jap_dictionary.trimDictionary()
		jap_dictionary.saveTrimmedDictionary()
	else:
		jap_dictionary.readTrimmedKatarigusa()



if __name__ == '__main__':
	main()