import mysql.connector
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from db_config import db_config

mydb = mysql.connector.connect(
    host=db_config["host"],
    port=db_config["port"],
    user=db_config["user"],
    password=db_config["password"],
    database=db_config["database"]
)

app = FastAPI()

@app.get("/cosine_similarity_matrices/{lesson_content_id}")
async def get_cosine_similarity_matrices(lesson_content_id: int):

    # Create a cursor object
    mycursor = mydb.cursor()

    # Query the database for the BLOB data
    query = f"SELECT id, html_content, lesson_content_id, student_first_name, student_last_name FROM lms.student_assignment WHERE lesson_content_id = {lesson_content_id}"
    mycursor.execute(query)
    results = mycursor.fetchall()

    if not results:
        return {"message": "No data found for this lesson_content_id."}
    else:
        # Create a list to hold the student names
        student_names = []
        # Create a dictionary to hold the cosine similarity matrices
        cosine_similarity_matrices = {}

        for result in results:
            id = result[0]
            blob_data = result[1]
            lesson_content_id = result[2]
            student_first_name = result[3]
            student_last_name = result[4]

            student_name = f"{student_first_name} {student_last_name}"
            student_names.append(student_name)

            if blob_data is None:  # Skip records with no BLOB data
                continue

            # Convert blob data to string and split into sentences
            blob_data_str = blob_data.decode('utf-8')
            sentences = blob_data_str.split('.')

            # Initialize TfidfVectorizer
            vectorizer = TfidfVectorizer()
            # Convert sentences to TF-IDF vectors
            tfidf_vectors = vectorizer.fit_transform(sentences)
            # Compute cosine similarity matrix
            cosine_similarity_matrix = cosine_similarity(tfidf_vectors)

            # Save cosine similarity matrix
            cosine_similarity_matrices[student_name] = cosine_similarity_matrix

        # Create a DataFrame to hold the cosine similarity matrices
        df = pd.DataFrame(columns=['student_names']+student_names)

        for i, name1 in enumerate(student_names):
            row = [name1]
            for j, name2 in enumerate(student_names):
                if i == j:
                    row.append('-')
                else:
                    cosine_similarity_matrix = cosine_similarity_matrices[name1]
                    cosine_similarity_score = cosine_similarity_matrix[j][0]
                    row.append(round(cosine_similarity_score*100, 2))
                    #row.append(cosine_similarity_score*100)
            df.loc[i] = row
        
        # Save the DataFrame to a CSV file in the same folder
        df.to_csv(f"{lesson_content_id}_lesson_plagiarism.csv", index=False)

        # Clean up
        mycursor.close()

        # Convert DataFrame to JSON and return it
        return df.to_json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8085)
