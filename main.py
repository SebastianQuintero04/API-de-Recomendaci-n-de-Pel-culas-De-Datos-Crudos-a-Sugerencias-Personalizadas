from fastapi import FastAPI
import pandas as pd
from IPython.display import display, Markdown
from collections import defaultdict
from fastapi.responses import HTMLResponse
import markdown
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import HashingVectorizer



app = FastAPI()

# Cargar los DataFrames
df = pd.read_csv('CleanData/movies_final.csv')
df_c = pd.read_csv('CleanData/creditos_final.csv')
df_highly_rated=pd.read_csv('DataML/movies_ml.csv')

@app.get("/")
def read_root():
    return {"message": "Bienvenido a mi proyecto MLOps/ Sebastian Quintero"}

# 1. Cantidad de filmaciones por mes
@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6, 
        'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    
    df['release_date'] = pd.to_datetime(df['release_date'])
    peliculas_mes = df[df['release_date'].dt.month == meses[mes.lower()]]
    cantidad = len(peliculas_mes)
    
    return {"mensaje": f"Cantidad de películas estrenadas en el mes de {mes}: {cantidad}"}

# 2. Cantidad de filmaciones por día
@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    dias = {
        'lunes': 'Monday', 'martes': 'Tuesday', 'miercoles': 'Wednesday', 
        'jueves': 'Thursday', 'viernes': 'Friday', 'sabado': 'Saturday', 'domingo': 'Sunday'
    }
    
    df['release_date'] = pd.to_datetime(df['release_date'])
    peliculas_dia = df[df['release_date'].dt.day_name() == dias[dia.lower()]]
    cantidad = len(peliculas_dia)
    
    return {"mensaje": f"Cantidad de películas estrenadas en los días {dia}: {cantidad}"}

# 3. Score de una filmación
@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    titulo_lower = titulo.lower()
    pelicula = df[df['title'].str.lower() == titulo_lower]
    
    if pelicula.empty:
        return {"mensaje": f"No se encontró la película '{titulo}'"}
    
    pelicula = pelicula.iloc[0]
    return {
        "mensaje": f"La película {pelicula['title']} fue estrenada en el año {pd.to_datetime(pelicula['release_date']).year} con un score/popularidad de {pelicula['popularity']}"
    }

# 4. Votos de un título
@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    titulo_lower = titulo.lower()
    pelicula = df[df['title'].str.lower() == titulo_lower]
    
    if pelicula.empty:
        return {"mensaje": f"No se encontró la película '{titulo}'"}
    
    pelicula = pelicula.iloc[0]
    if pelicula['vote_count'] >= 2000:
        return {
            "mensaje": f"La película {pelicula['title']} fue estrenada en el año {pd.to_datetime(pelicula['release_date']).year}. La misma cuenta con un total de {pelicula['vote_count']} valoraciones, con un promedio de {pelicula['vote_average']}"
        }
    else:
        return {"mensaje": f"La película {pelicula['title']} no cumple con la condición de tener al menos 2000 valoraciones."}

# 5. Éxito de un actor
@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    mask = df_c['cast'].str.contains(nombre_actor, case=False, na=False)
    peliculas_actor = df.loc[mask]
    
    cantidad_peliculas = len(peliculas_actor)
    retorno_total = peliculas_actor['return'].sum()
    retorno_promedio = retorno_total / cantidad_peliculas if cantidad_peliculas > 0 else 0
    
    return {
        "mensaje": f"El actor {nombre_actor} ha participado de {cantidad_peliculas} cantidad de filmaciones, el mismo ha conseguido un retorno de {retorno_total:.2f} con un promedio de {retorno_promedio:.2f} por filmación"
    }

# 6. Éxito de un director
@app.get("/get_director/{nombre_director}")
def get_director_info(nombre_director: str):
    mask = df_c['directors'].str.contains(nombre_director, case=False, na=False)
    peliculas_director = df.loc[mask]
    
    retorno_total = peliculas_director['return'].sum()
    mensaje = f"El director {nombre_director} ha conseguido un retorno total de {retorno_total:.2f}. Películas:\n\n"
    
    peliculas_agrupadas = defaultdict(list)
    for _, pelicula in peliculas_director.iterrows():
        peliculas_agrupadas[pelicula['title']].append(pelicula)
    
    for titulo, peliculas in peliculas_agrupadas.items():
        pelicula = peliculas[0]
        mensaje += f"- **{titulo}**:\n"
        mensaje += f"  * Fecha de lanzamiento: {pelicula['release_date']}\n"
        mensaje += f"  * Retorno: {pelicula['return']:.2f}\n"
        mensaje += f"  * Costo: {pelicula['budget']:.2f}\n"
        mensaje += f"  * Ganancia: {pelicula['revenue'] - pelicula['budget']:.2f}\n"
        if len(peliculas) > 1:
            mensaje += f"  * Apariciones: {len(peliculas)}\n"
        mensaje += "\n"
    
    return Markdown(mensaje)

# Importamos las librerías para calcular la similitud del coseno y para vectorizar el texto de las características.
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import HashingVectorizer

# Aseguramos que los datos de la columna 'overview' sean strings
df_highly_rated['overview'] = df_highly_rated['overview'].fillna('').astype('str')

# Aseguramos que los datos de la columna 'genres' sean strings
df_highly_rated['genre'] = df_highly_rated['genre'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else '')

# Reemplazamos los valores NaN con cadenas vacías en la columna 'production_companies'
df_highly_rated['company'] = df_highly_rated['company'].fillna('')

# Convertimos la columna 'production_companies' a string si es necesario
df_highly_rated['company'] = df_highly_rated['company'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x)

# Creamos una nueva columna combinando las características de interés
df_highly_rated['combined_features'] = df_highly_rated['overview'] + ' ' + df_highly_rated['genre'] + ' ' + df_highly_rated['company']

# Convertimos todos los textos a minusculas para evitar duplicados
df_highly_rated['combined_features'] = df_highly_rated['combined_features'].str.lower()

# Inicializamos el HashingVectorizer para vectorizar el texto en una matriz de características. La dimensión de la matriz se establece en 2000.
hash_vectorizer = HashingVectorizer(stop_words='english', n_features=2000)

# Transformamos los datos
hash_matrix = hash_vectorizer.fit_transform(df_highly_rated['combined_features'])

# Calculamos la similitud del coseno de la matriz de características usando la función cosine_similarity()
cosine_sim = cosine_similarity(hash_matrix)

# Creamos un índice de películas utilizando los títulos de las películas como clave y los índices como valores.
indices = pd.Series(df_highly_rated.index, index=df_highly_rated['title']).drop_duplicates()

@app.get("/recomendacion/{titulo}")
def recomendacion(titulo):
    '''Ingresas un nombre de pelicula y te recomienda 5 similares
    '''
    if titulo not in df_highly_rated['title'].values:
        return 'La película no se encuentra en el conjunto de datos de muestra.'
    else:
        # Obtenemos el índice de la película que coincide con el título
        idx = indices[titulo]

        # Obtenemos las puntuaciones de similitud de todas las películas con la puntuación de la película dada
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Ordenamos las películas en función de las puntuaciones de similitud
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Obtenemos las puntuaciones de las 5 películas más similares
        sim_scores = sim_scores[1:6]

        # Obtenemos los índices de las películas
        movie_indices = [i[0] for i in sim_scores]

        # Devolvemos las 5 películas más similares
        return {'lista recomendada': df_highly_rated['title'].iloc[movie_indices].tolist()}
