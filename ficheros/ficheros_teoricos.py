
# Date: 06/April/2023 - Thursday
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


# def input(f=open(".ficheros_In1.txt")): return f.readline().rstrip() # uncomment for debugging


import os

def main():
	
	line1 = "Es mi primera línea"
	line2 = "Es mi segunda línea"
	line3 = "Es mi tercera línea"
	line4 = "Es el final del fichero!"

	# 1
	name = open("primerfichero.txt","w")
	name.close()

	os.system("ls")
	
	# 2
	with open("segundofichero.txt","w") as segundo:
		pass
	os.system("ls")

	# 3
	file1 = open("primerfichero.txt","a")
	file1.write(line1)
	file1.close()
	# print the file
	print("file 1:")
	for line in open("primerfichero.txt","r"):
		print(line)

	# 4
	file2 = open("segundofichero.txt","a")
	file2.write(line2 + "\n" )
	file2.write(line3 + "\n" )
	file2.write(line4 + "\n" )
	file1 = open("primerfichero.txt","r")
	file1_str = file1.read()
	file2.write(file1_str)
	file2.close()

	print("file2:")
	# print the following file
	print(open("segundofichero.txt","r").read())


if __name__ == '__main__':
	main()

