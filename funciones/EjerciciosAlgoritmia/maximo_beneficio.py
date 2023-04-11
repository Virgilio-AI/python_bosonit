
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


# def input(f=open(".maximo_beneficio_In1.txt")): return f.readline().rstrip() # uncomment for debugging


def main():
	lst = list(map(int,input().split(',')))
	max_right = [0]* len(lst)
	max_right[-1] = lst[-1]
	for i in range(len(lst)-2,-1,-1):
		max_right[i] = max(max_right[i+1], lst[i])

	ans = 0 
	for i in range(len(lst)):
		ans = max(ans, max_right[i] - lst[i])
	print(ans)


if __name__ == '__main__':
	main()

