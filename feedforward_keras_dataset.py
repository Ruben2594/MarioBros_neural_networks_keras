import time
import numpy as np
import h5py
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist
from numpy import genfromtxt


class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)
	#plot_losses(self.losses)


def load_data():
    print 'Cargando mario.csv...'
    mario_level_data = genfromtxt('mario.csv', delimiter=',')
    #data_x = entrada, data_y = salida
    data_x = mario_level_data[:,:172032]
    data_y = mario_level_data[:,[172032,172033,172034,172035,172036,172037]]
    split_number = len(data_x)/2
    (X_train, X_test) = data_x[:split_number,:], data_x[split_number:,:]
    (y_train, y_test) = data_y[:split_number,:], data_y[split_number:,:]
    #Convertir imagenes en un vector de pixeles
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    #Los valores van de 0 a 255 en una escala de grises (normalizar datos de entrada)
    #convertir de 0-255 a 0-1
    X_train /= 255
    X_test /= 255

    print 'Data cargada...'
    #init_model()
    return [X_train, X_test, y_train, y_test]
    


def init_model():
    start_time = time.time()
    print 'Modelo de aprendizaje... '
    model = Sequential() #secuencia de capas
    model.add(Dense(500, input_dim=172032)) #datos de entrada 172031
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(6)) #salida de 6 dimensiones 
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    print 'Model compield in {0} seconds'.format(time.time() - start_time)
    #run_network()
    return model


def run_network(data=None, model=None, epochs=20, batch=256):
    try:
        start_time = time.time()
        if data is None:
            X_train, X_test, y_train, y_test = load_data()
        else:
            X_train, X_test, y_train, y_test = data

        if model is None:
            model = init_model()

        history = LossHistory()

        print 'Entrenamiento del modelo...'
        model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch,
                  callbacks=[history],
                  validation_data=(X_test, y_test), verbose=2)

        print "Training duration : {0}".format(time.time() - start_time)
        score = model.evaluate(X_test, y_test, batch_size=16)
	print len(X_test[0])
        result = model.predict(X_test, batch_size=16, verbose=1)
	print(result)
	#Error
	plot_losses(history.losses)
	print ("%.2f% %" % (100-score[1]*100))
        print "Network's test score [loss, accuracy]: {0}".format(score)
	#Guardar el modelo...
	print 'Guardando modelo en el disco...'
	model_json = model.to_json()
	# serialize model to JSON
	with open("model.json", "w") as json_file:
    		json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk H5")

        return model, history.losses
    except KeyboardInterrupt:
        print ' KeyboardInterrupt'
        return model, history.losses


def plot_losses(losses):
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(losses)
    #ax.set_title('Loss per batch')
    #fig.show()
    #err = (losses,accuracy)
    #errors.append(err)
    plt.plot(losses)
    plt.show()

if __name__ == '__main__':
    load_data()
    init_model()
    run_network()
    #plot_losses
