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


# =============
# ==== diapositiva 1 =====
# =============

# Función 1
def showAnimal(name, n_legs):
	print("El animal", name, "tiene", n_legs, "patas.")

# Función 2
def printArgs(*args):
	print("Se han recibido", len(args), "argumentos.")

# Función 3
def sumAndSubtract(a, b):
	return a + b, a - b

# Función 4
def multiply(a, b):
	return a * b

# Función 5
def modulo(a, b):
	return a % b

# Función 6
def calculate(func, a, b):
	return func(a, b)

# Función 7
def printNameAndEmail(name, email="Sin determinar"):
	print("El nombre es", name, "y el correo electrónico es", email)

# Función 8
def sumUntilZero(n):
	total = 0
	for i in range(n + 1):
		total += i
	return total

# =============
# ==== slide two =====
# =============

# Función 1
square = lambda x: x ** 2

# Función 2
greaterThan999 = lambda x: True if x ** 2 > 999 else False

# Función 3
multiply = lambda x, y: x * y

# Función 4
words = ["perro", "gato", "loro", "mono", "oso"]
sortedWords = sorted(words, key=lambda x: x[1])



# Función 1
def doubleList(lst):
	return [2*x for x in lst]

# Función 2
def squareEven(lst):
	return [x**2 if x%2==0 else x for x in lst]

# Función 3
def sumLists(lst1, lst2):
	return [x+y for x,y in zip(lst1, lst2)]

# Función 4
def countAs(lst):
	return [x.count('a') for x in lst]

# Función 5
def getNegatives(lst):
	return [x for x in lst if x < 0]

# Función 6
def getVowels(string):
	return [x for x in string if x in "aeiouAEIOU"]

# Función 7
def multiplyList(lst):
	result = 1
	for x in lst:
		result *= x
	return result

# Función 8
import re
def sumNumbers(string):
	nums = [ int(x) for x in re.findall('\d+',string) ]
	return sum(nums)





def main():
	# slide one
	showAnimal("Perro", 4)
	
	printArgs(1, 2, 3)
	
	sum, sub = sumAndSubtract(5, 3)
	print("La suma es", sum, "y la resta es", sub)
	
	result = calculate(multiply, 4, 5)
	print("El resultado de la multiplicación es", result)
	
	result = calculate(modulo, 20, 7)
	print("El módulo es", result)
	
	printNameAndEmail("Juan", "juan@example.com")
	printNameAndEmail("Ana")
	
	total = sumUntilZero(5)
	print("La suma hasta 0 es", total)


	## slide two

	# Ejemplo de uso de las funciones
	print(square(5))
	
	print(greaterThan999(10))
	print(greaterThan999(20))
	
	print(multiply(3, 5))
	
	print(sortedWords)



	# Ejemplo de uso de las funciones
	lst1 = [1, 2, 3, 4, 5]
	print(doubleList(lst1))
	
	lst2 = [1, 2, 3, 4, 5, 6]
	print(squareEven(lst2))
	
	lst3 = [1, 2, 3, 4, 5]
	lst4 = [10, 20, 30, 40, 50]
	print(sumLists(lst3, lst4))
	
	lst5 = ["casa", "carro", "perro", "avión"]
	print(countAs(lst5))
	
	lst6 = [-1, 2, -3, 4, -5]
	print(getNegatives(lst6))
	
	string1 = "Hola mundo"
	print(getVowels(string1))
	
	lst7 = [1, 2, 3, 4, 5]
	print(multiplyList(lst7))
	
	string2 = "Hoy es 5 de abril y tengo 30 años"
	print(sumNumbers(string2))




if __name__ == '__main__':
	main()

