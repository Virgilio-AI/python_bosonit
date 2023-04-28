
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
	# operador aritmetico
	a = 50
	b = 5.0
	c = 100
	d = a + b
	e = a - b
	f = d * e
	g = f / a
	h = f % a
	print("a =", a)
	print("b =", b)
	print("c =", c)
	print("d =", d)
	print("e =", e)
	print("f =", f)
	print("g =", g)
	print("h =", h)





	# operador de comparacion
	a = 50
	b = 10
	print(a == b)
	print(a != b)
	print(a > b)
	print(a <= b)  # Salida: False




	# operador de asignacion
	y = 999
	y += 1
	y -= 10
	y *= 10
	y /= 5
	y %= 60
	print(y)  # Salida: 27.0








	# operadores logicos
	a = 10
	b = 100
	c = 200
	d = 300
	print(a > b and c < d)
	print((a + b >= c) or (b + c >= d))




	# operadores de pertinencia
	list1 = [1, 2, 3, 4, 5]
	list2 = ['Hello', 'Word', 'Python']
	list3 = ['Operator', 'Membership', 100, 200]
	print(5 in list1)
	print("Hello" in list2 and "Python" in list2)
	print(list2 == list3)  # Salida: False





	# operadores bit a bit
	a = 1
	b = 2
	c = a & b
	print(c)
	d = a ^ b
	print(d)
	e = a << b
	print(e)






	# ejercicio con operadores de identidad
	a = 3
	b = 3.0
	print(type(a) == int)
	print(type(b) == bool)
	print(type(b) == float)



if __name__ == '__main__':
	main()

