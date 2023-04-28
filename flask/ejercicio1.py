from flask import Flask

app = Flask(__name__)

@app.route('/sumar/<int:num1>/<int:num2>')
def sumar(num1, num2):
	suma = num1 + num2
	resultado = f"La suma de {num1} y {num2} es: {suma}"
	return resultado

if __name__ == '__main__':
	app.run(debug=True)

