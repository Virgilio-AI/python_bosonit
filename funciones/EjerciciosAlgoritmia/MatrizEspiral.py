
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


# def input(f=open(".MatrizEspiral_In1.txt")): return f.readline().rstrip() # uncomment for debugging


def main():
	m = int(input())
	n = int(input())
	# read  the given matrix from the input
	matrix = []
	for i in range(m):
		matrix.append(list(map(int, input().split())))



	i,j = 0,0
	k = 0
	ans_mat = [[0 for i in range(n)] for j in range(m)]
	ans_mat[i][j] = 1
	ans = []
	ans.append(matrix[i][j])
	while k < n*m - 1:
		if j+1 < len(ans_mat[0]) and ans_mat[i][j+1] == 0:
			j += 1
			k += 1
		elif i+1 < len(ans_mat) and ans_mat[i+1][j] == 0:
			i += 1
			k += 1
		elif j - 1 >= 0 and ans_mat[i][j-1] == 0:
			j -= 1
			k += 1
		elif i - 1 >= 0 and ans_mat[i-1][j] == 0:
			i -= 1
			k += 1
		ans_mat[i][j] = 1
		ans.append(matrix[i][j])
	print(ans)













if __name__ == '__main__':
	main()

