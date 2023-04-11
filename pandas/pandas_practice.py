
# Date: 06/April/2023 - Thursday
# Author: Virgilio Murillo Ochoa
# personal github: Virgilio-AI
# linkedin: https://www.linkedin.com/in/virgilio-murillo-ochoa-b29b59203
# contact: virgiliomurilloochoa1@gmail.com
# web: virgiliomurillo.com

from typing import *
import heapq as hp
from collections import deque
from collections import defaultdict
import sys

# dic = defaultdict(int)
# set = set() # .remove(val),.add(val),.discard(val),.pop(),.clear()
# dic = {} # .remove(id),dic[key] = val, dic.get(key,0)
# arr = [] # .append(val),.pop(),.remove(val),.sort(),.reverse(),.insert(pos,val),.clear()


# def input(f=open(".pandas_practice_In1.txt")): return f.readline().rstrip() # uncomment for debugging

import pandas as pd
import numpy as np

def main():
	# Definimos las listas
	#1
	brand = ['Camp', 'Camp', 'Petzl', 'Petzl', 'Edelrid', 'Edelrid', 'Edelrid', 'Black Diamond', 'Black Diamond', 'Mammut']
	models = ['Energy', 'Jasper', 'Simba', 'Adjama New', 'Moe', 'Orion', 'Leaf', 'Xenes', 'Chaos', 'Ophir']
	prices = [39.90, 56.00, 45.00, 75.00, 49.90, 99.90, 65.00, 119.90, 99.90, 55.00]
	
	#2
	# Creamos la Serie
	s = pd.Series(data=prices, index=models)
	print(s)

	#3
	print(s[s < 70])

	#4
	# aplica 10% de descuento cuando brand es 'Edelrid'
	news = s.copy()
	news[news.index == 'Edelrid'] *= 0.9
	
	print(news)

	
	# last diapositive
	#1
	df = pd.DataFrame({'brand':brand, 'model':models, 'price':prices})
	print(df)
	#2
	print(df.brand)
	#3
	print(df.iloc[5])
	#4
	df[df.model == "Energy"] = 9999
	#5
	df['Discount'] = 50
	print(df)
	#6
	newdf = df.copy()
	newdf['Sales'] = 100
	newdf['Total'] = newdf['price'] * newdf['Sales']
	print(newdf)
	#7
	# borra la primera y la séptima fila
	newdf = newdf.drop([0,7])
	print(newdf)

	#8
	cols = np.array(newdf.columns)
	drop = cols[[1,3]]
	newdf = newdf.drop(columns = list(drop))
	print(newdf)

	#9
	print(newdf[newdf.brand == "Edelrid"].info())

	#10
	print(newdf[newdf.price > 70])

	#11
	sales = np.random.randint(0,500,10)
	df['Sales'] = sales

	#12
	df['Total'] = df['price'] * df['Sales']
	print(df)

	#13
	print(df.sort_values(by=['Sales'], ascending=False))

	#14
	# Usando la función apply cambia el formato a dos decimales
	df['price'] = df['price'].apply(lambda x: round(x,2))
	print(df)

	#15
	print(df.describe())


	#16
	print(df.describe(include=['object']))

	#17
	print(df.isnull().sum())

	#18
	# calcula el numero de productos para cada marca
	df.groupby('brand').agg({'model':'count'})

	#19
	df.to_excel('pandas_practice.xlsx', index=False)









if __name__ == '__main__':
	main()

