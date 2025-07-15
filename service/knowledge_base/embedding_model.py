from FlagEmbedding import FlagAutoModel

class EmbeddingModel:
    def __init__(self, model_name, model_type):
        self.model_name = model_name
        self.use_fp16 = model_type
        self.model = FlagAutoModel.from_finetuned(model_name, use_fp16=self.use_fp16)
        
    def get_embedding(self, text):
        return self.model.encode(text)

Embedding_model = EmbeddingModel("BAAI/bge-base-en-v1.5", True)

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = FlagAutoModel.from_finetuned("BAAI/bge-base-en-v1.5", use_fp16=True)
    return _embedder