import pandas as pd
import numpy as np
from dotenv import load_dotenv
from humanfriendly.terminal import output

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr



books= pd.read_csv("books_with_emotions.csv")


books["large_thumbnail"]= books["thumbnail"]+"&fife=w800"
books["large_thumbnail"]= np.where(
    books["large_thumbnail"].isna(),
    "cover_not_found_image",
    books["large_thumbnail"]
)

raw_documents= TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter= CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
documents= text_splitter.split_documents(raw_documents)

embeddings= HuggingFaceEmbeddings(
    model_name= "sentence-transformers/all-MiniLM-L6-v2"
)
db_books= Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")


def retrieve_semantic_recommendations(
        query: str,
        category:str=None,
        tone: str= None,
        initial_top_k: int= 50,
        final_top_k: int= 16
)->pd.DataFrame:
    recs=db_books.similarity_search_with_score(query, k=initial_top_k)
    book_list= [(int(rec[0].page_content.strip('"').split()[0])) for rec in recs]

    book_recs= books[books["isbn13"].isin(book_list)].head(final_top_k)
    tone= tone.lower()

    if category != "All":
        book_recs= book_recs[book_recs["simple_categories"]==category].head(final_top_k)
    else:
        book_recs= book_recs.head(final_top_k)

    if tone=="happy":
        book_recs.sort_values(by= "joy", ascending=False, inplace=True)
    elif tone=="sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)
    elif tone=="angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone=="surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone=="suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query=query, category=category, tone=tone)
    results= []
    for _, row in recommendations.iterrows():
        description= row["description"]
        truncated_desc_split= description.split()
        truncated_desc= " ".join(truncated_desc_split[:30]) + "..."

        authors_split= row["authors"].split(";")
        if len(authors_split) ==2:
            authors_str= f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str= f"{ ' '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str= row["authors"]

        caption= f"{row["title"]} by {authors_str} : {truncated_desc}"
        results.append((row["large_thumbnail"], caption))
    return results


categories= ["All"] + sorted(books["simple_categories"].unique())
tones= ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks() as dashboard:
    gr.Markdown("# Semantic book Recommender")

    with gr.Row():
        user_query= gr.Textbox(label= "Please enter a description of a book:",
                               placeholder="e.g, A story about happiness")
        category_dropdown= gr.Dropdown(choices=categories, label= "Please select a category:",value= "All")
        tone_dropdown= gr.Dropdown(choices=tones, label= "Please select an emotional tone:",value= "All")
        submit_button= gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    outputs= gr.Gallery(label= "Recommended books", columns= 8, rows= 2)

    submit_button.click(fn= recommend_books,
                        inputs= [user_query, category_dropdown, tone_dropdown],
                        outputs= outputs)

    if __name__=="__main__":
        dashboard.launch(theme= gr.themes.Glass())