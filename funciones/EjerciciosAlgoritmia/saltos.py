
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


# def input(f=open(".saltos_In1.txt")): return f.readline().rstrip() # uncomment for debugging


def main():
	lst = list(map(int,input().split()))
	ans = 1
	dic = {}
	dic[0] = [0,lst[0]]
	dic[1] = [1,lst[1]]

	for i in range(2,len(lst)):
		rem1 = (dic[ans-1][0] + dic[ans-1][1]) - i 
		rem2 = (dic[ans][0] + dic[ans][1]) - i
		if rem2 < lst[i]:
			dic[ans] = [i,lst[i]]

		if rem1 == -1:
			ans += 1
			dic[ans] = [i,lst[i]]

	print(ans)








if __name__ == '__main__':
	main()

