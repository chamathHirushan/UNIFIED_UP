import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# Include the class definition exactly as it was when pickled
class UnifiedQAWithRetrieval:
    def __init__(self, model, tokenizer, embed_model):
        self.model = model
        self.tokenizer = tokenizer
        self.embed_model = embed_model

    @staticmethod
    def extract_after_first_newline(text: str):
        parts = text.split("\n", 1)
        return parts[0].strip(), parts[1].strip()

    def retrieve_relevant_chunk(self, question, context, max_tokens=500):
        paragraphs = [p.strip() for p in context.split("\n") if p.strip()]
        paragraph_embeddings = self.embed_model.encode(paragraphs, convert_to_tensor=True)
        question_embedding = self.embed_model.encode(question, convert_to_tensor=True)

        similarities = util.cos_sim(question_embedding, paragraph_embeddings)[0]
        ranked_idx = similarities.argsort(descending=True)

        selected_text = ""
        token_count = 0
        for idx in ranked_idx:
            para = paragraphs[idx]
            para_tokens = len(para.split())
            if token_count + para_tokens <= max_tokens:
                selected_text += para + " "
                token_count += para_tokens
            if token_count >= max_tokens:
                break

        return selected_text.strip()

    def answer_question(self, input_text, **generate_kwargs):
        question, context = self.extract_after_first_newline(input_text)
        retrieved_context = self.retrieve_relevant_chunk(question, context)
        final_input = question + " \n " + retrieved_context

        inputs = self.tokenizer(final_input, return_tensors="pt", truncation=True, padding=True)
        output_ids = self.model.generate(**inputs, **generate_kwargs)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


# Load the QA system
with open("unifiedqa_with_retrieval.pkl", "rb") as f:
    qa_system = pickle.load(f)

# Example question and context
input_text = """Who discovered penicillin?
Penicillin was discovered by Alexander Fleming in 1928. 
It was later developed into a medicine by Howard Florey and Ernst Chain.
"""

# Generate an answer
answer = qa_system.answer_question(input_text, max_length=64)
print("Answer:", answer)