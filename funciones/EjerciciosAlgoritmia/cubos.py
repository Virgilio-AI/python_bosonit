
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


# def input(f=open(".cubos_In1.txt")): return f.readline().rstrip() # uncomment for debugging


def main():
	arr = [int(x) for x in input().split()]
	ans = 0
	maxi = -1
	for val in arr:
		if val > maxi:
			maxi = val
	for val in arr:
		ans += maxi - val
	print(ans)



if __name__ == '__main__':
	main()

