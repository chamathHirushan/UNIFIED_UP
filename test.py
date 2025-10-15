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
input_text = """What instrument did Tolkien play?
    J.R.R. Tolkien was born in Bloemfontein, South Africa in 1892.
    He is best known as the author of high fantasy works like The Hobbit and The Lord of the Rings.
    Tolkien's family moved to England when he was a child, after his father's death.
    He studied at Exeter College, Oxford, where he focused on English language and literature.
    During World War I, Tolkien served in the British Army as a second lieutenant.
    After the war, he worked on the Oxford English Dictionary and began his academic career.
    Tolkien was a professor of Anglo-Saxon at Oxford University for many years.
    He was part of a literary discussion group called The Inklings, which included C.S. Lewis.
    The Hobbit was published in 1937 and was an immediate success.
    The Lord of the Rings was published in three volumes between 1954 and 1955.
    Tolkien's works have inspired countless adaptations, including major motion pictures.
    He was also a skilled artist and created many illustrations for his Middle-earth stories.
    Tolkien passed away in 1973, but his literary legacy continues to influence fantasy literature.
    The violin is a string instrument that is played with a bow.
    It has four strings tuned in perfect fifths and is the smallest and highest-pitched instrument in its family.
    J.R.R. Tolkien was known to play the violin during his school years.
    The violin originated in Italy in the early 16th century.
    """

# Generate an answer
answer = qa_system.answer_question(input_text, max_length=64)
print("Answer:", answer)