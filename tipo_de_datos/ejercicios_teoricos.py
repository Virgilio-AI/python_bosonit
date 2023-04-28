
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


# def input(f=open(".ejercicios_teoricos_In1.txt")): return f.readline().rstrip() # uncomment for debugging


def main():
	# Ejercicios numericos
	x = 10
	y = 3.14
	z = 2 + 3j
	print(z.real)
	print(z.imag)


	# ejercicios boolean
	a = True
	b = False
	c = bool(1)
	print(a)
	print(b)
	print(c)




	# Ejercicios con string
	s = " Master Python "
	print(len(s))
	print(s[0])
	print(s[-2])
	s = s.lstrip()
	print(s)
	print(s[1::2])
	s = s.lower()
	print(s)
	s_split = s.split()
	print(s_split)
	s_replace = s.replace("python", "JAVA")
	print(s_replace)


	# Ejercicios con listas
	lista1 = [10, 20, 'Hello', 20.5]
	lista1.append("Word")
	lista1.insert(0, "Python")
	del lista1[1]
	lista2 = [20, 40, 'Bye']
	lista_final = lista1 + lista2
	print(lista_final)


	# Ejercicios con set
	conjunto1 = {"coche", "motocicleta", "bicicleta"}
	conjunto1.add("avión")
	conjunto1.remove("coche")
	conjunto2 = set(["avión", "coche", "tractor"])
	conjunto3 = conjunto1.intersection(conjunto2)
	conjunto4 = conjunto1.union(conjunto2)
	print(conjunto4)

	# dictionary
	diccionario1 = dict(nombre="Juan", edad=25, coche=5000)
	diccionario2 = {"nombre": "Pedro", "edad": 30, "coche": 10000}
	claves = ["nombre", "edad", "coche"]
	valores = ["María", 20, 8000]
	diccionario3 = dict(zip(claves, valores))
	print(diccionario1.values())
	print(diccionario2.values())
	print(diccionario3.values())
	print(diccionario1.keys())
	print(diccionario2.keys())
	print(diccionario3.keys())
	print(diccionario1["coche"])
	print(diccionario2["coche"])
	print(diccionario3["coche"])
	diccionario1["avión"] = 100
	print(diccionario1.items())


if __name__ == '__main__':
	main()

