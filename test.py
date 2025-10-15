import pickle

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