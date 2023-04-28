
# Date: 17/April/2023 - Monday
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


# def input(f=open(".main_In1.txt")): return f.readline().rstrip() # uncomment for debugging


def main():
	# ejercicios con if
	a = 10
	b = 100
	if a == 10:
		print("Es igual a 10")
	else:
		print("Es diferente de 10")
	if a == 10 and b == 10:
		print("Todos son igual a 10")
	elif a == 10:
		print("Sólo a es igual a 10")
	elif b == 10:
		print("B es igual a 10")
	else:
		print("Ninguno es igual a 10")
	if a % 2 != 0 and b % 2 != 0:
		print("A y B son impares")
	else:
		print("A y B no son impares")



	# ejercicios usando while 
	i = 1
	while i == 1:
		print(i)
		i = i - 1
	i = 1
	while True:
		print(i)
		i = i - 1
		if i != 1:
			break
	i = 0
	while i < 10:
		i = i + 1
		if i == 6:
			print("Ejecución terminada")
			break



	# ejercicios operador de asignación
	a = ['Hello', 'World']
	b = ['Python', 3.9]
	c = 'HelloWorldPython'
	for char in c:
		print(char)
	for val in a:
		print(val)
	for val_a, val_b in zip(a, b):
		print(val_a, val_b)
	for idx, val in enumerate(b):
		print(f"Índice: {idx}, Valor: {val}")



if __name__ == '__main__':
	main()

