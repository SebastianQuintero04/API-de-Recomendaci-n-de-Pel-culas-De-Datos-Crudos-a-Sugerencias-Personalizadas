from fastapi import FastAPI
import pandas as pd
from sqlalchemy import create_engine, text
from collections import defaultdict
from fastapi.responses import HTMLResponse
import markdown
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import HashingVectorizer

app = FastAPI()

# Crear una conexión a la base de datos
engine = create_engine('sqlite:///movies_database.db')
df_highly_rated = pd.read_csv('DataML/movies_ml.csv')

@app.get("/")
def read_root():
    return {"message": "Bienvenido a mi proyecto MLOps/ Sebastian Quintero"}

@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    meses = {
        'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04', 'mayo': '05', 'junio': '06', 
        'julio': '07', 'agosto': '08', 'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
    }
    
    query = text(f"""
    SELECT COUNT(*) as cantidad
    FROM movies_final
    WHERE strftime('%m', release_date) = :mes
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {"mes": meses[mes.lower()]}).fetchone()
    
    cantidad = result[0]
    
    return {"mensaje": f"Cantidad de películas estrenadas en el mes de {mes}: {cantidad}"}

@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    dias = {
        'lunes': 'Monday', 'martes': 'Tuesday', 'miércoles': 'Wednesday', 
        'jueves': 'Thursday', 'viernes': 'Friday', 'sábado': 'Saturday', 'domingo': 'Sunday'
    }
    
    query = text(f"""
    SELECT COUNT(*) as cantidad
    FROM movies_final
    WHERE strftime('%w', release_date) = :dia
    """)
    
    dia_num = {'Monday': '1', 'Tuesday': '2', 'Wednesday': '3', 'Thursday': '4', 'Friday': '5', 'Saturday': '6', 'Sunday': '0'}[dias[dia.lower()]]
    
    with engine.connect() as conn:
        result = conn.execute(query, {"dia": dia_num}).fetchone()
    
    cantidad = result[0]
    
    return {"mensaje": f"Cantidad de películas estrenadas en los días {dia}: {cantidad}"}

@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    query = text("""
    SELECT title, release_date, popularity
    FROM movies_final
    WHERE LOWER(title) = LOWER(:titulo)
    LIMIT 1
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {"titulo": titulo}).fetchone()
    
    if result is None:
        return {"mensaje": f"No se encontró la película '{titulo}'"}
    
    return {
        "mensaje": f"La película {result.title} fue estrenada en el año {result.release_date[:4]} con un score/popularidad de {result.popularity}"
    }

@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    query = text("""
    SELECT title, release_date, vote_count, vote_average
    FROM movies_final
    WHERE LOWER(title) = LOWER(:titulo)
    LIMIT 1
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {"titulo": titulo}).fetchone()
    
    if result is None:
        return {"mensaje": f"No se encontró la película '{titulo}'"}
    
    if result.vote_count >= 2000:
        return {
            "mensaje": f"La película {result.title} fue estrenada en el año {result.release_date[:4]}. La misma cuenta con un total de {result.vote_count} valoraciones, con un promedio de {result.vote_average}"
        }
    else:
        return {"mensaje": f"La película {result.title} no cumple con la condición de tener al menos 2000 valoraciones."}

@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    query = text("""
    SELECT m.title, m.retorno
    FROM movies_final m
    JOIN creditos_final c ON m.id = c.id
    WHERE c.cast LIKE :actor
    """)
    
    with engine.connect() as conn:
        results = conn.execute(query, {"actor": f"%{nombre_actor}%"}).fetchall()
    
    cantidad_peliculas = len(results)
    retorno_total = sum(r.retorno for r in results)
    retorno_promedio = retorno_total / cantidad_peliculas if cantidad_peliculas > 0 else 0
    
    return {
        "mensaje": f"El actor {nombre_actor} ha participado de {cantidad_peliculas} cantidad de filmaciones, el mismo ha conseguido un retorno de {retorno_total:.2f} con un promedio de {retorno_promedio:.2f} por filmación"
    }

@app.get("/get_director/{nombre_director}")
def get_director_info(nombre_director: str):
    query = text("""
    SELECT m.title, m.release_date, m.retorno, m.budget, m.revenue
    FROM movies_final m
    JOIN creditos_final c ON m.id = c.id
    WHERE c.directors LIKE :director
    """)
    
    with engine.connect() as conn:
        results = conn.execute(query, {"director": f"%{nombre_director}%"}).fetchall()
    
    retorno_total = sum(r.retorno for r in results)
    mensaje = f"El director {nombre_director} ha conseguido un retorno total de {retorno_total:.2f}. Películas:\n\n"
    
    peliculas_agrupadas = defaultdict(list)
    for r in results:
        peliculas_agrupadas[r.title].append(r)
    
    for titulo, peliculas in peliculas_agrupadas.items():
        pelicula = peliculas[0]
        mensaje += f"- **{titulo}**:\n"
        mensaje += f"  * Fecha de lanzamiento: {pelicula.release_date}\n"
        mensaje += f"  * Retorno: {pelicula.retorno:.2f}\n"
        mensaje += f"  * Costo: {pelicula.budget:.2f}\n"
        mensaje += f"  * Ganancia: {pelicula.revenue - pelicula.budget:.2f}\n"
        if len(peliculas) > 1:
            mensaje += f"  * Apariciones: {len(peliculas)}\n"
        mensaje += "\n"
    
    return {"mensaje": mensaje}

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

