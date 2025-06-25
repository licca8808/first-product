from flask import Blueprint, render_template, request # type: ignore
from .model import model, data

main = Blueprint("main", __name__)

@main.route("/", methods=["GET","POST"])
def index():
    return render_template("index.html")

@main.route("/answer", methods=["GET","POST"])
def answer():
    user_input = request.form["query"]
    sentences = data["説明"].tolist()
    sentences.append(user_input)

    sentence_embeddings = model.encode(sentences)

    from scipy.spatial import distance # type: ignore
    distances = distance.cdist(
        [sentence_embeddings[-1]], sentence_embeddings, metric="cosine"
    )[0]

    results = list(enumerate(distances))
    results = sorted(results, key=lambda x: x[1])
    
    top_n = 3
    top_recommendations = []
    for idx, dist in results[1:top_n + 1]:
        row = data.iloc[idx]
        top_recommendations.append({
            "場所": row["場所"],
            "キャッチコピー": row["キャッチコピー"],
            "説明": row["説明"]
        })


    return render_template("answer.html", recommendation_list=top_recommendations, user_input=user_input)
