# Open AI Gym Atari Pong game :ping_pong:

_Desarrollo de una inteligencia artificial entrenada para jugar pong de atari, implementado con kera's y gym._

## Implementación e instalación 
Para este proyecto utilizamos la libreria de Open AI Gym environment. Para el entrenamiento de la inteligencia artificial, se utiliza el modelo de red neuronal convolucional usando Keras.


### Instalación 📋

_Para la instalación del proyecto es necesario descargar algunas librerias que el programa pide para su correcto funcionamiento._
_Siga en orden los pasos para instalar correctamente las librerias que a continuación se muestran:_

Una vez abierto el proyecto, porfavor abra la terminal para comenzar la instalación.
_Escriba en orden y ejecutando una por una las siguientes librerías_

Preferentemente se sugiere que trabaje en un entorno virtual. (Si no desea trabajar el proyecto en un entorno virtual puede saltar este paso).
```
$ pvirtualenv venv
$ source ./venv/Scripts/activate
```

```
$ pip install pandas seaborn scikit-learn

$ pip install opencv-python

$ pip install gym
```

_En caso de que no detecte que gym esté instalado, puede intentar ejecutar alguno de estos comandos:_
```
$ pip install -e .[all]

$ pip install gym[all]
```


## Correr el programa 🚀

_Una vez instalado las librerias necesarias necesitamos ejecutar el programa, esto se divide en dos partes:_

_1:_ Ejecutar para entrenar. 
Para entrenar nuestra inteligencia artificial es necesario primero correr el archivo **main.py**, para eso ejecutamos en la terminal el siguiente comando:

```
$ python main.py 
```

El programa comenzará ejecutar mostrandonos los episodios, el tiempo que tomó terminarlo, el puntaje de la IA y el puntaje maximo en el episodio.

Por defecto y cuestión de rendimiento el programa al momento de entrenar la AI no ejecuta el emulador, pero si desea ver el emulador en proceso mientras se entrena el programa solamente cambie en la linea 26 del archivo **main.py** la palabra al final por un **TRUE**

_Como se muestra a continuación:_

```
for i in range(1000000):
    timesteps = agent.total_timesteps
    timee = time.time()
    score = environment.play_episode(name, env, agent, debug = TRUE) 
    scores.append(score)
    if score > max_score:
        max_score = score
```

Cada vez que se ejecuten 50 episodios, el programa guardará los datos que fueron entrenados y luego ejecutará otros 50 y así sucesivamente.
_Es recomendable dejar entrenando la IA por lo menos 24 horas para que aprenda a jugar al 100%._


_2:_ Ejecutar verificar el entrenamiento

Una vez terminando el entrenamiento se puede ejecutar el archivo **debug.py** para mostrar el avance optenido.
Esto se puede ejecutar desde la terminal de la siguiente forma:

```
$ python debug.py 
```

Si la inteligencia artificial fue entrenada correctamente (Dejandola entrenando minimo 50 episodios). 
El programa se ejecuta y muestra el emulador de open AI Gym de la siguiente forma:


https://user-images.githubusercontent.com/55809555/144955802-757b6107-0d30-4fb0-8204-9d9ac4f28970.mp4

_Ejemplo de una IA entrenada 100 episodios._



Notas a tomar en cuenta:
* _Considere que entre menos tiempo se deje entrenando a la inteligencia artificial menos aprende, es decir, entre más tiempo se entrene más aprende a jugar correctamente._
* _Si su computador tiene menos de 8 ram, es rencomendable no dejar ejecutando el programa a más de 50 episodios._
* _Verifique que su computador pueda soportar eficientementen el emulador._
* _Si al momento de entrenar, el emulador o los episodios tardan mucho en completarse, es recomendable tener deshabilitado el emulador de open AI Gym al momento de entrenar._
* _No se asuste si al momento de ejecutar la IA entrenada su emulador va algo lento._



## Explicación de los archivos implementados  📄

A continuación se explica de manera general el funcionamiento de los archivos implentados en este proyecto.

_Cada uno de los archivos tienen notas en forma de comentario explicando de manera breve el funcionamiento de estos._


### preprocess_frame.py

Para tener una mejor idea del entrenamiento que le hemos dado a nuestra inteligencia artificial es necesario implementar un buen entorno de trabajo para que esta se pueda ejecutar sin problema y dando un buen rendimiento, por lo que coemenzamos el documento configuarando el emulador para darnos una mejor vista de lo que está pasando.


### agent_memory.py

El ajente de memoria guarda la información del entrenamiento como experiencias para el agente, es decir, cuando estamos entrenando realmente estamos ejecutando diferentes posibilidades que podrían pasar para que nuestro agente logre tocar 1 vez la pelota, y cuando lo logra esa información se guarda como una experiencia para que en la siguiente ocasión de juego el agente contemple que en cierto lugar logro pegarle una vez a esa pelota.

Toma los registros de éxito y fracaso para que el agente pueda eprender basando en la "experiencia".


### environment.py

En este archivo manejamos 4 diferentes funciones que permiten que el espacio de trabajo se ejecute. Lo que hace es ejecutar de manera correcta estas funciones para emular un entorno de aprendizaje para el agente. Se implementa como y con que va a trabajar el agente para ejecutar correctamente el entrenamiento.

* _La función make_env ():_ Establece el enviroment o entorno de trabajo nuevo en el que se va ejecutar el programa.
* _la función initialize_new_game ():_ Esta función inicia un nuevo espacio de juego para que no se sobre escriba los datos de anteriores ejecuciones.
* _La función take_step ():_ Toma una serie de pasos para implementar la formula matematica con el que se calcula el aprendizaje del agente, se toma en cuenta diferentes acciones de ejecución para el agente, el cual permiten que pueda aprender correcamente.
* _La función play_episode():_ Por ultimo se ejecuta de nuevo un nuevo juego a partir de 0 y al final devuelve la puntuación que se acumuló durante el episodio.


### the_agent.py

Para entrenar el agente es necesario darle a conocer como debe de trabajar, por lo que, en este archivo se implenta el modelo de red neuronal convolucional para entrenar el agente. Construye la red neuronal y toma pequeñas muestras para validar que esta funcione correctamente. 

Genera por así decirlo su propio espacio de entrenamiento para probar la red neuronal con números aleatorios dirigidos en un intervalos.

![modelo](https://user-images.githubusercontent.com/55809555/144963408-6f273d55-6bb7-48ff-a126-0cbcd3ae8740.png)

_Formula implementada como función._


### main.py

Usamos este archivo simplemente para ejecutar los anteriores vistos, inicia el agente y manda a llamar las funciones ya validadas de los archivos anteriores para ejecutar correctamente el modelo de entremiento. Muestra los datos del modelo en proceso como el puntaje, tiempo de duración del episodio, los episodios, etc.

En este apartado se ejecuta el agente para comenzar a entrenarlo.


### debug.py

Comienza la ejecución del modelo ya entrenado, mostrando de manera más clara y precisa como ha aprendido nuestra inteligencia artificial.

_En este archivo se ejecuta el emulador open AI Gym para enseñar el funcionamiento final de la IA._


## Fuentes y referencias :fountain: 

* Basado en el repositorio de [Meredevs](https://github.com/Meredevs/DQN_Pong)
* Página de referencia en [towardsdatascience](https://towardsdatascience.com/getting-an-ai-to-play-atari-pong-with-deep-reinforcement-learning-47b0c56e78ae)
* Pagina de [Gym](https://gym.openai.com)





