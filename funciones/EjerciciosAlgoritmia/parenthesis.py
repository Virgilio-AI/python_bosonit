
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


# def input(f=open(".parenthesis_In1.txt")): return f.readline().rstrip() # uncomment for debugging


def main():
	stack = []
	stri = input()
	for i in stri:
		if i == '(' or i == '{' or i == '[':
			stack.append(i)
		else:
			condition = True
			condition &= len(stack) == 0
			condition &= i == ')' and stack[-1] != '('
			condition &= i == '}' and stack[-1] != '{'
			condition &= i == ']' and stack[-1] != '['
			if condition:
				print("False")
				return
			stack.pop()
	if len(stack) == 0:
		print("True")



if __name__ == '__main__':
	main()

