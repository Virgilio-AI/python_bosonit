
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


# def input(f=open(".numpy_In1.txt")): return f.readline().rstrip() # uncomment for debugging

import numpy as np

def main():
	# define ndarray with values 5,8,9
	#1
	arr = np.array([5,8,9])
	#2
	print(arr.size)
	#3
	print(type(arr))
	#4
	arr3d = np.array([[5],[8],[9]])
	#5
	# Creamos el primer ndarray con valores de 25
	a = np.full((3, 3), 0)
	b = np.full((3,3),1)
	c = np.empty((3,3))
	print(a)
	print(b)
	print(c)

	# second diapositive

	#1
	arr1 = np.array([[1,2],[3,6]])
	print(arr1)
	#2
	arr2 = np.array([[6,6],[7,7]])
	print(arr2)
	#3
	# multiplicacion de matrices
	print(arr1*arr2)
	#4
	suma = arr1+arr2
	print(suma)
	#5
	# suma 999 a cada elemento de la matriz arr1
	print(arr1 + 999)

	# tercer diapositiva

	#1
	arr1 = np.array([100,1,1001,123])
	#2
	print(arr1)
	#3
	print(arr1[0])
	#4
	mat4 = np.array([[1,1,1,1],[2,2,2,2]])
	#5
	print(mat4[0][0])
	print(mat4[0][1])
	#6
	# imprime el valor de la primera fila y la primera columna
	print("primera file y primera columna")
	print(mat4[0,0:4])
	print(mat4[0:2,0])

	#7
	# imprime los valores en mat4 mayores a 1
	print(mat4[mat4 > 1])

	# ====== diapositiva
	
	#1
	narray = np.array([100,999,998,997,997])
	#2
	# find the value 997
	print(np.where(narray == 997))
	#3
	# modifica todos los valores menores de 999 a 9999
	narray[narray < 999] = 9999

	print(narray)

	#4
	# obten los valores unicos de narray
	print(np.unique(narray))

	#5
	# return True if all the values are greater than 1000
	print(np.all(narray > 1000))

	#6
	# devuelve True si algun valor es mayor que 1000
	print(np.any(narray > 1000))

	# ordena el array en forma ascendente
	print(np.sort(narray))






if __name__ == '__main__':
	main()

