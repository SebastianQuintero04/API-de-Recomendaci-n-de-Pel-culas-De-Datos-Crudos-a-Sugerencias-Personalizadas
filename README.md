# Sistema de Recomendación de Películas

![Portada](./assets/portada.png)

## Índice
- [Introducción](#introducción)
- [Metodología](#metodología)
  - [Extracción, Transformación y Carga (ETL)](#extracción-transformación-y-carga-etl)
  - [Análisis Exploratorio de Datos (EDA)](#análisis-exploratorio-de-datos-eda)
  - [Desarrollo de API](#desarrollo-de-api)
  - [Sistema de Recomendación](#sistema-de-recomendación)
- [Instalación y Uso](#instalación-y-uso)
- [API Endpoints](#api-endpoints)
- [Demostración de la API](#demostración-de-la-api)
- [Deployment](#deployment)
- [Resultados y Conclusiones](#resultados-y-conclusiones)
- [Autor](#autor)

## Introducción

Este proyecto tiene como objetivo desarrollar un sistema de recomendación de películas para una start-up que provee servicios de agregación de plataformas de streaming. El sistema incluye un proceso de ETL para limpiar y preparar los datos, un análisis exploratorio de datos (EDA), una API para consultar información sobre películas y actores, y un modelo de machine learning para recomendar películas similares.

## Metodología

### Extracción, Transformación y Carga (ETL)

Se realizó un proceso de ETL para limpiar y preparar los datos. Esto incluyó desanidar campos, rellenar valores nulos, estandarizar formatos de fecha, crear nuevas columnas calculadas y eliminar columnas innecesarias. El proceso detallado se encuentra en [`ETL.ipynb`](./ETL.ipynb).

### Análisis Exploratorio de Datos (EDA)

Se llevó a cabo un análisis exploratorio de los datos para investigar relaciones entre variables, identificar outliers y patrones interesantes. Se utilizaron visualizaciones como nubes de palabras para entender mejor los datos. El análisis completo está en [`EDA.ipynb`](./EDA.ipynb).

### Desarrollo de API

Se desarrolló una API utilizando FastAPI para proporcionar acceso a los datos procesados. La API incluye endpoints para consultar información sobre películas, actores y directores. El código de la API se encuentra en [`main.py`](./main.py).

### Sistema de Recomendación

Se implementó un sistema de recomendación de películas basado en similitud. El sistema recomienda 5 películas similares a una película dada. Los detalles de implementación están en [`sistema.ipynb`](./sistema.ipynb).

## Instalación y Uso

1. Clona este repositorio
2. Instala las dependencias: `pip install -r requirements.txt`
3. Ejecuta la API: `uvicorn main:app --reload`

## API Endpoints

- `/cantidad_filmaciones_mes/{mes}`: Retorna la cantidad de películas estrenadas en un mes dado.
- `/cantidad_filmaciones_dia/{dia}`: Retorna la cantidad de películas estrenadas en un día de la semana dado.
- `/score_titulo/{titulo}`: Retorna el score/popularidad de una película dada.
- `/votos_titulo/{titulo}`: Retorna la cantidad de votos y el promedio de votaciones de una película.
- `/get_actor/{nombre_actor}`: Retorna información sobre un actor dado.
- `/get_director/{nombre_director}`: Retorna información sobre un director dado.
- `/recomendacion/{titulo}`: Retorna una lista de 5 películas recomendadas similares a la película dada.

## Demostración de la API

Para ver una demostración del funcionamiento de la API, por favor visite el siguiente enlace:

[Demostración de la API](https://drive.google.com/drive/folders/1ZypFQD_QvqfLNlweDm5bN-qOxoSpL5jv?usp=sharing)

En este enlace encontrará videos y capturas de pantalla que muestran cómo utilizar los diferentes endpoints de la API y los resultados que se obtienen.

## Deployment

La API ha sido desplegada y está disponible para su uso. Puede acceder a la documentación interactiva de la API y probar sus funcionalidades en el siguiente enlace:

[API de Recomendación de Películas](https://api-de-recomendaci-n-de-pel-culas-de.onrender.com/docs)

Esta interfaz le permitirá explorar todos los endpoints disponibles, ver sus descripciones y probarlos directamente desde su navegador.

## Resultados y Conclusiones

[Aquí puedes incluir un resumen de los principales hallazgos, insights o conclusiones del proyecto]

## Autor

Este proyecto fue desarrollado por Sebastian Quintero Hernandez 

- LinkedIn: [www.linkedin.com/in/sebastian-quintero0413](https://www.linkedin.com/in/sebastian-quintero0413)
- Correo electrónico: [quinterosebastian820@gmail.com](mailto:quinterosebastian820@gmail.com)
