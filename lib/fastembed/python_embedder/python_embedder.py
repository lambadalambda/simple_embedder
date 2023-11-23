from erlport.erlterms import Atom
from fastembed.embedding import FlagEmbedding as Embedding

class Embedder:
    def __init__(self):
        self.model = None

    def load_model(self, name):
        self.model = Embedding(model_name=name, max_length=512)
        return Atom(b"ok")

embedder = Embedder()

def load_model(name):
    name = name.decode("utf-8")
    return embedder.load_model(name)

def get_text_embedding(text):
    text = text.decode('utf-8')
    [embedding] = list(embedder.model.embed([text]))
    return embedding.tolist()

def get_query_embedding(text):
    text = text.decode('utf-8')
    [embedding] = list(embedder.model.query_embed([text]))
    return embedding.tolist()

def get_passage_embedding(text):
    text = text.decode('utf-8')
    [embedding] = list(embedder.model.passage_embed([text]))
    return embedding.tolist()
