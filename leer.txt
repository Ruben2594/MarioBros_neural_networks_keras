Laboratorio 1 de Inteligencia Artificial

Primero, instalar
sudo apt-get install fceux

De no hacerlo mario no se instalara

Crear el mundo de mario
1- Crear carpeta de mundo de mario llamado lab_1
2- Desde la consola ingresar crear un ambiente virtual para el mundo de mario
virtualenv -p `which python2` lab_1env
3- Acceder al ambiente virtual
source lab_1env/bin/activate
4-- Instalar la librerias para el sistema de entrenamiento
pip install gym
pip install numpy
pip install panda
pip install gym-pull
pip install matplotlib
pip install keras
5- Probar la activacion del mundo de mario, creando archivo basico
editar un archivo llamado mario.py
