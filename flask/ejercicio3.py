from flask import Flask, request

app = Flask(__name__)

usuarios = ['Juan', 'María', 'Pedro', 'Ana', 'Luisa']

@app.route('/buscar_usuario')
def buscar_usuario():
	texto = request.args.get('texto')
	if texto:
		count = sum(1 for usuario in usuarios if texto.lower() in usuario.lower())
		mensaje = f"Se encontraron {count} usuarios que coinciden con la búsqueda: {texto}"
		return mensaje
	else:
		return "Error: No se proporcionó el parámetro ?texto= en la URL."

if __name__ == '__main__':
	app.run(debug=True)
