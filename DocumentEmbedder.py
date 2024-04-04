import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSeq2SeqLM
import json
import numpy as np

class Score:
    """
    Score offers a collection of static methods for extracting specific scores from a tensor of logits,
    typically obtained as an output from models like T5 and flan-T5. These scores are computed based on
    predefined token IDs corresponding to 'true' and 'false' representations in the T5 and flan-T5 vocabularies.

    Methods include:
    - extra_id: Extracts the logit corresponding to token ID 32089.
    - difference: Calculates the difference between the logits for the 'true' (token ID 1176) and 'false' (token ID 6136) tokens.
    - softmax: Computes the softmax scores for the 'true' (token ID 1176) and 'false' (token ID 6136) logits, and returns the softmax value for the 'true' token.

    Note:
    - Token ID 1176 corresponds to the 'true' token in the T5 and flan-T5 vocabularies.
    - Token ID 6136 corresponds to the 'false' token in the T5 and flan-T5 vocabularies.
    - Token ID 32089 corresponds to the '<extra_id_10>' token in the T5 and flan-T5 vocabularies.
    """
    
    @staticmethod
    def extra_id(logits):
        return logits[:, 32089]
    
    @staticmethod
    def difference(logits):
        true_logits = logits[:, 1176]
        false_logits = logits[:, 6136]
        return true_logits - false_logits
    
    @staticmethod
    def softmax(logits):
        true_logits = logits[:, 1176]
        false_logits = logits[:, 6136]
        scores = [torch.nn.functional.softmax(torch.stack([true_logit, false_logit]), dim=0)[0]
                 for true_logit, false_logit in zip(true_logits, false_logits)]
        return torch.stack(scores)


class DocumentEmbedder:
    def __init__(
        self, 
        json_file_path=None,
        data=None, 
        context_tokenizer='facebook/dragon-plus-context-encoder', 
        query_tokenizer='facebook/dragon-plus-query-encoder', 
        context_model='facebook/dragon-plus-context-encoder', 
        query_model='facebook/dragon-plus-query-encoder', 
        rerank_model='castorini/monot5-base-msmarco-10k', 
        rerank_tokenizer='castorini/monot5-base-msmarco-10k',
        device=torch.device("cuda")
    ):
        if json_file_path is not None and data is not None:
            raise ValueError("Both json_file_path and data cannot be provided simultaneously. Choose one.")

        self.json_file_path = json_file_path
        self.data = data   
        
        # Download the models using the provided or default checkpoints
        self.model = {
            'tokenizer': AutoTokenizer.from_pretrained(context_tokenizer),
            'con_model': AutoModel.from_pretrained(context_model).to(device),
            'query_model': AutoModel.from_pretrained(query_model).to(device),
            'rerank_model': AutoModelForSeq2SeqLM.from_pretrained(rerank_model).to(device),
            'rerank_tokenizer': AutoTokenizer.from_pretrained(rerank_tokenizer),
            'device': device
        }

        # Embed the data during initialization
        if self.json_file_path is not None:
            with open(self.json_file_path, 'r') as json_file:
                self.data = json.load(json_file)
        elif self.data is not None:
            self.data = self.data
        else:
            raise ValueError("Either json_file_path or data must be provided.")

        self._embed_data_(self.data)

    def _embed_data_(self, data):
        embeddings = []

        for i in tqdm(range(0, len(data), 128), desc="Embedding Documents"):
            batch_data = data[i:i + 128]
            batch_texts = [doc['text'] for doc in batch_data]
            ctx_input = self.model['tokenizer'](batch_texts, padding="max_length", max_length=512, truncation=True, return_tensors='pt')
            batch_input_ids = ctx_input.input_ids.to(self.model['device'])
            batch_attention_mask = ctx_input.attention_mask.to(self.model['device'])
            with torch.no_grad():
                batch_embedding = self.model['con_model'](
                    batch_input_ids,
                    attention_mask=batch_attention_mask
                ).last_hidden_state[:, 0, :]
            batch_embedding = batch_embedding.cpu()
            embeddings.append(batch_embedding)

        embeddings = torch.cat(embeddings)

        # Update documents with embeddings
        for i, doc in enumerate(data):
            doc['embedding'] = embeddings[i].tolist()

    def retrieve(self, query, top_k=100, return_scores=True):
        query_input = self.model['tokenizer'](query, max_length=512, padding="max_length", truncation=True, return_tensors='pt').to(self.model['device'])
        with torch.no_grad():
            query_emb = self.model['query_model'](
                query_input.input_ids.to(self.model['device']),
                attention_mask=query_input.attention_mask.to(self.model['device'])
            ).last_hidden_state[:, 0, :]
        scores = [(query_emb @ torch.tensor(doc['embedding']).T.to(self.model['device'])).cpu().item() for doc in self.data]
        top_k_indices = np.argsort(scores)[-top_k:][::-1]
        if return_scores:
            top_k_documents = [{"text": self.data[i]['text'], "retrieve_score": scores[i]} for i in top_k_indices]
        else:
            top_k_documents = [self.data[i]['text'] for i in top_k_indices]

        return top_k_documents

    def rerank(self, query, documents, n_docs=2):
        reranked_documents = []
        rerank_model = self.model['rerank_model']
        rerank_tokenizer = self.model['rerank_tokenizer']
        rerank_model.eval()

        for document in documents:
            data = document['text']
            input_text = f"Query: {query} Document: {data} Relevant: "
            features = rerank_tokenizer(
                input_text, 
                truncation=True, 
                return_tensors="pt", 
                max_length=500, 
                padding=True,
            ).to(self.model['device'])
            input_ids = features.input_ids
            attention_mask = features.attention_mask
            decode_ids = torch.full(
                (input_ids.size(0), 1),
                rerank_model.config.decoder_start_token_id,
                dtype=torch.long
            ).to(self.model['device'])

            with torch.no_grad():
                output = rerank_model(
                    input_ids=input_ids.to(self.model['device']),
                    attention_mask=attention_mask.to(self.model['device']),
                    decoder_input_ids=decode_ids.to(self.model['device']),
                )
            logits = output.logits[:, 0, :]
            get_score = getattr(Score, 'softmax')
            rerank_score = get_score(logits).item()
            document["rerank_score"] = rerank_score
            reranked_documents.append(document)

        reranked_documents = sorted(reranked_documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_documents[:n_docs]

    def get_relevant_docs(self, batch_size, query, top_k=100, return_scores=True):
        relevant_docs = self.retrieve(query, top_k, return_scores)
        reranked_docs = self.rerank(query, relevant_docs)
        return reranked_docs

