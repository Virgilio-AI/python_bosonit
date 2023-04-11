
# Date: 05/April/2023 - Wednesday
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


# def input(f=open(".clases_y_objetos_In1.txt")): return f.readline().rstrip() # uncomment for debugging

import math
class FirstExercise():
	def __init__(self,number,chapter):
		self.number = number
		self.chatper = chapter
	def print_number(self):
		print(self.number)

class Circle():
	def __init__(self,radio):
		self.radio = radio
		self.area = math.pi * (radio ** 2)
		self.perimetro = 2 * math.pi * radio
	def get_area(self):
		return self.area
	def get_perimetro(self):
		return self.perimetro
	def modificar_radio(self,radio):
		self.radio = radio

class Vehicle():
	owner = "Bosonit"
	def __init__(self,nombre,velocidad_maxima, kilometraje):
		self.nombre = nombre
		self.velocidad_maxima = velocidad_maxima
		self.kilometraje = kilometraje

	def print_capacity(self,capacidad):
		self.capacidad = capacidad
		print(f" la capacidad de {self.nombre} es {capacidad}")



class Bus(Vehicle):
	def print_capacity(self,capacidad = 50):
		print(f" la capacidad de {self.nombre} es {capacidad}")

def main():
	obj = FirstExercise(6,"Clases y objetos")
	bus = Bus('mercedes',100,200000)
	print(bus.print_capacity())
	print(bus.owner)




if __name__ == '__main__':
	main()

