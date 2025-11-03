import pickle

# Ruta del archivo .pkl
archivo = '../tu_archivo.pkl'

# Abrimos el archivo en modo lectura binaria
with open(archivo, 'rb') as f:
    contenido = pickle.load(f)  # Deserializamos el objeto

# Ahora podemos imprimir o explorar el contenido
print(contenido)

