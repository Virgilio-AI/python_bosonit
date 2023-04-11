
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


# def input(f=open(".Cables_In1.txt")): return f.readline().rstrip() # uncomment for debugging


def main():
	# si el numero de salidas hembra es diferente al numero de salidas macho, no se puede conectar
	line = input()
	m,h = 0,0
	for ch in line:
		if ch == 'H':
			h+=1
		elif ch == 'M':
			m += 1
	if m != h:
		print("false")
	else:
		print("true")




if __name__ == '__main__':
	main()
