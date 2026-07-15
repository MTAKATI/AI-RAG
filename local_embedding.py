import torch
from Vector_index import VectorIndex
from transformers import AutoTokenizer, AutoModel

class LocalEmbedding:

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", distance_metric="cosine"):
        """
        Initializes the tokenizer, transformer model, and vector store.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[LocalEmbedding] loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("[LocalEmbedding Model is ready!]")

        self.store = VectorIndex(distance_metric=distance_metric, embedding_fn=self._embed_one)

    ###
    # Private Helpers
    ###
    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs mean pooling on token embeddings accounting for attention masks.
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
        token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / token_counts

    def _embed_one(self, text: str) -> list[float]:
        """
        Helper method to extract a single embedding vector as a list of floats.
        """
        return self.get_embeddings([text])[0].tolist()
    
    ###
    # Public API
    ###
    def get_embeddings(self, text: list[str]) -> torch.Tensor:
        """
        Generates normalized embeddings for a list of strings.
        """
        encoded = self.tokenizer(
            text,                  # Fixed: Changed from `texts` to `text`
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",   # Fixed: Changed from `return_tensor` to `return_tensors`
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.model(**encoded)

        embeddings = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu()
    
    def build_index(self, chunks: list[str]) -> None:
        """
        Embeds chunks of text and indexes them into the vector store.
        """
        embeddings = self.get_embeddings(chunks)
        for chunk, embedding in zip(chunks, embeddings):
            self.store.add_vector(embedding.tolist(), {"content": chunk})
        print(f"[LocalEmbedding] Indexed {len(chunks)} chunks.")

    def search(self, question: str, k: int = 3) -> list[tuple[dict, float]]:
        """
        Searches the vector store for the closest matching chunks.
        """
        # Fixed: Corrected parameter typing and variable name mismatch
        return self.store.search(question, k=k)
    
    def get_content(self, question: str, k: int = 3) -> str:
        """
        Retrieves the content of top-k documents formatted as a single markdown string.
        """
        results = self.search(question, k=k)
        return "\n\n---\n\n".join(doc["content"] for doc, _ in results)