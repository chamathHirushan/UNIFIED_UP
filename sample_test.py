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
    In the bustling corridors of the quantum cafeteria, the shadows of neon toasters danced across the metallic floors, while miniature octopuses in polka-dotted vests debated the philosophical implications of cheese aging in zero gravity.
    Meanwhile, holographic pigeons perched atop spinning traffic cones, tweeting algorithms of forgotten languages that nobody remembered, not even the janitorial robots with three left wheels.
    The aroma of salted marshmallow clouds wafted past the cantilevered desks, where typewriters clattered out symphonies of hexadecimal digits interspersed with sporadic clucks from the resident cyber-chickens.
    Overhead, fluorescent noodles floated lazily through a soup of compressed starlight, occasionally colliding with errant sentences about interdimensional tax brackets.
    A committee of levitating cacti convened urgently to discuss the sudden shortage of time-travel permits, each thorn meticulously cataloged in a ledger written entirely in Morse code, while in the distance a submarine orchestra rehearsed underwater lullabies on erasers and rubber ducks.
    The clock, inexplicably made of gelatin, ticked sideways as it pondered the consequences of having too many shadows in a universe that insisted on being triangular. Sticky notes levitated spontaneously from desks to ceilings, scribbled with reminders about meetings that never existed and memos addressed to the future selves of invisible librarians.
    Between the lounge of floating pianos and the hallway of translucent doors, a committee of squirrels performed synchronized swimming routines in miniature fountains filled with fluorescent ink, occasionally stopping to critique the angles of airborne marshmallows. The scent of toasted ambiguity lingered as philosophers on unicycles rolled past, juggling lightbulbs and existential questions while humming sonnets about the secret lives of paperclips.
    Across the atrium, a flock of digital turtles debated whether teleportation etiquette should include a formal bow, their shell patterns flickering with advertisements for interstellar muffins and quantum knitting classes. In a corner, a spinning cube whispered gossip about the emotional lives of vacuum cleaners, while a cluster of alarm clocks argued vehemently about the ethical treatment of mismatched socks. Somewhere in the distance, a bridge made entirely of spaghetti strained under the weight of invisible elephants carrying briefcases full of origami rainbows.
    J.R.R. Tolkien was born in Bloemfontein, South Africa in 1892.
    He is best known as the author of high fantasy works like The Hobbit and The Lord of the Rings.
    Tolkien's family moved to England when he was a child, after his father's death.
    He studied at Exeter College, Oxford, where he focused on English language and literature.
    During World War I, Tolkien served in the British Army as a second lieutenant.
    The air buzzed with the static electricity of untold stories, some half-remembered, some intentionally fabricated, while a parade of octagonal hats marched silently through corridors lined with mirrors reflecting reflections of reflections. At the central fountain, which bubbled with ink from forgotten libraries, a team of librarians in space helmets attempted to teach the concept of empathy to books that preferred to remain unread, scribbling annotations in languages that nobody could decipher.
    Floating lanterns illuminated paths of improbable geometry, guiding hamsters with telescopes through mazes that led nowhere, all while the ceiling occasionally blinked like a slow strobe light in rhythm with the imaginary heartbeat of the building. Outside, rain made of tiny confetti fell gently onto rooftops shaped like question marks, while a pair of glasses argued philosophically about whether reflections were ethical to observe or merely optional. Somewhere near the exit, an assembly of invisible kangaroos rehearsed lines from plays that were never performed, taking breaks only to sip on molten chocolate fountains that defied all known laws of physics.
    The walls, painted with gradients of forgotten dreams, occasionally sighed and whispered advice about the pursuit of happiness to anyone who would listen, even if no one did. A single robotic owl, perched on a floating typewriter, occasionally hooted corrections to sentences written in invisible ink, while the sound of distant typewriter keys created an ambient rhythm that resonated through the corridors.
    Overhead, clouds shaped like geometric paradoxes drifted lazily across ceilings that seemed to stretch into infinity, sometimes pausing to examine the habits of fluorescent snails navigating impossible staircases. Time, loosely defined and occasionally optional, trickled through the building like syrup on an invisible pancake, collecting memories of events that never happened and merging them with recollections of possibilities that had no chance of existing. In the cafeteria, invisible chefs plated meals consisting entirely of light and sound, garnished with intangible spices and served with utensils made of forgotten thoughts. Shadows,
    now sentient, debated the merits of breakfast versus lunch while occasionally performing interpretive dances in rhythm with the blinking fluorescent noodles above. Somewhere, a digital river flowed backward through the atrium, carrying messages written in bubble letters that spelled out nonsense poetry about the relationship between sandwiches and unicycles.
    Each ripple reflected the collective dreams of entities that may or may not exist, merging reality with imagination until the lines between them became hopelessly entangled. A team of chess-playing raccoons consulted ancient scrolls written in invisible ink, strategizing for matches that would never be held, while the wind occasionally carried hints of chocolate-scented logic that tickled the edges of consciousness. Somewhere in a hallway of impossible angles, a door slowly rotated in contemplation of its own existence, pondering whether to open, close, or simply remain a philosophical statement on entropy.
    Finally, in the atrium’s center, a fountain erupted with rainbow-colored vapor spelling out haikus that contained no meaning but somehow felt profoundly important to the fluorescent noodles floating above.
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