
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


# def input(f=open(".lluvia_In1.txt")): return f.readline().rstrip() # uncomment for debugging


def main():
	elements = list(map(int, input().split(',')))
	maxleft = [0] * len(elements)
	maxright = [0] * len(elements)
	maxleft[0] = elements[0]
	maxright[-1] = elements[-1]

	for i in range(1,len(elements)):
		maxleft[i] = max(maxleft[i-1],elements[i])

	for i in range(len(elements) -2, -1, -1):
		maxright[i] = max(maxright[i+1],elements[i])
	ans = 0
	for i in range(len(elements)):
		ans += min(maxleft[i],maxright[i]) - elements[i]
	print(ans)






if __name__ == '__main__':
	main()

