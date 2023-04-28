
from flask import Flask

app = Flask(__name__)

@app.route('/usuario/<string:nombre>/<string:apellido>')
def crear_usuario(nombre, apellido):
	if isinstance(nombre, str) and isinstance(apellido, str):
		username = nombre[0].lower() + apellido.lower()
		return username
	else:
		return "Error: Los parámetros deben ser strings."

if __name__ == '__main__':
	app.run(debug=True)
