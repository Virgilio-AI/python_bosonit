
# Date: 11/April/2023 - Tuesday
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


# def input(f=open(".escaleras_In1.txt")): return f.readline().rstrip() # uncomment for debugging


def main():
	val = int(input())
	cache = [-1] * (val + 1)
	def numero_formas(n):
		if n == 1:
			return 1
		if n == 2:
			return 2
		if cache[n] != -1:
			return cache[n]
		cache[n] = numero_formas(n-1) + numero_formas(n-2)
		return cache[n]
	print(numero_formas(val))


if __name__ == '__main__':
	main()

