# import joblib
# import sys
# import logging
# import pandas as pd
# from pathlib import Path
# from typing import List
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from src.logger import create_log_path, CustomLogger


# ## Logging
# # Set logging path
# log_file_path = create_log_path('training_logger')
# # Create a custom logger
# training_logger = CustomLogger(logger_name='training_logger', log_filename=log_file_path)
# # Set logging level
# training_logger.set_log_level(level=logging.INFO)



# def load_dataframe(path):
#     df = pd.read_csv(path)
#     return df


# def similarity_fn(dataframe: pd.DataFrame, tfidf_vectorizer: TfidfVectorizer) -> List[float]:
#     """Calculate cosine similarity using TF-IDF."""
#     similarity = []
#     for _, row in dataframe.iterrows():
#         docs = (row['text1'], row['text2'])
#         tfidf_matrix = tfidf_vectorizer.transform(docs)
#         cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
#         similarity.append(cosine_sim)
#     return similarity


# def train_tfidf(data: pd.DataFrame) -> TfidfVectorizer:
#     """Train TF-IDF on both text1 and text2."""
#     tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
#     combined_texts = data["text1"].tolist() + data["text2"].tolist()
#     tfidf_vectorizer.fit(combined_texts)  # ✅ Train on both columns
#     training_logger.save_logs('Trained TF-IDF on text1 & text2')
#     return tfidf_vectorizer


# def save_model(model, save_path):
#     joblib.dump(value=model,
#                 filename=save_path)

# def main():
#     # get the current path
#     current_path = Path(__file__)
#     # go the the root path
#     root_path = current_path.parent.parent.parent
#     # input path 
#     input_path = root_path / 'data' / 'processed' / 'final'
#     # training dat path
#     training_data_path = input_path / sys.argv[1]
#     # load dataframe
#     train_df = load_dataframe(training_data_path)   
#     # save model path 
#     model_output_path = root_path  /'models' / 'models'
#     model_output_path.mkdir(exist_ok=True)
#     # Train the TF-IDF model on both text1 & text2
#     tfidf_vectorizer = train_tfidf(train_df)
#     # Save the trained model
#     save_model(tfidf_vectorizer, model_output_path / "tf-idf.joblib")


# if __name__ == '__main__':
#     main()



import joblib
import sys
import logging
import pandas as pd
from pathlib import Path
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.logger import create_log_path, CustomLogger

## Logging
log_file_path = create_log_path('training_logger')
training_logger = CustomLogger(logger_name='training_logger', log_filename=log_file_path)
training_logger.set_log_level(level=logging.INFO)


def load_dataframe(path: str) -> pd.DataFrame:
    """Load CSV data into a pandas DataFrame."""
    return pd.read_csv(path)


def similarity_fn(dataframe: pd.DataFrame, model: SentenceTransformer) -> List[float]:
    """Calculate cosine similarity using SentenceTransformer embeddings."""
    similarity_scores = []
    for _, row in dataframe.iterrows():
        embeddings = model.encode([row['text1'], row['text2']], convert_to_tensor=True)
        cosine_sim = cosine_similarity([embeddings[0].cpu().numpy()], [embeddings[1].cpu().numpy()])[0][0]
        similarity_scores.append(cosine_sim)
    return similarity_scores


def train_model() -> SentenceTransformer:
    """Load a pre-trained Sentence Transformer model."""
    model = SentenceTransformer("all-MiniLM-L6-v2")  # ✅ Using a lightweight BERT model
    training_logger.save_logs("Loaded pre-trained Sentence Transformer: all-MiniLM-L6-v2")
    return model


def save_model(model: SentenceTransformer, save_path: Path):
    """Save the trained Sentence Transformer model."""
    model.save(str(save_path))


def main():
    # Get the current path
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    input_path = root_path / "data" / "processed" / "final"
    
    # Load data
    training_data_path = input_path / sys.argv[1]
    train_df = load_dataframe(training_data_path)
    
    # Define model path
    model_output_path = root_path / "models" / "models" / "sentence_transformer"
    model_output_path.mkdir(exist_ok=True)

    # Load pre-trained Sentence Transformer
    model = train_model()

    # Save the model
    save_model(model, model_output_path)

    # Compute similarity scores
    train_df["similarity_score"] = similarity_fn(train_df, model)
    print(train_df[["text1", "text2", "similarity_score"]].head())  # ✅ Check similarity output


if __name__ == "__main__":
    main()
