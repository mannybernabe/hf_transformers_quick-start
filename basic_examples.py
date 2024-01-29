from transformers import pipeline
import torch

# Example 1 -- Sentiment analysis

classifier = pipeline("sentiment-analysis")

result = classifier("I've been waiting for a hugging face course!")

print(result)




# Example 2 -- Text generation

generator = pipeline("text-generation", model="distilgpt2")

result = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

print(result)



# Example 3 -- Zero shot classification
classifier = pipeline("zero-shot-classification")

result = classifier(
    "This is a course about Python list comprehension",
    candidate_labels=["education", "politics", "business"],
)

print(result)



# Example 4 -- Named Entity Recognition (NER)

# Load the NER pipeline
ner = pipeline("ner", grouped_entities=True)

# Example text
text = "Hugging Face is a technology company based in New York."

# Apply NER to the text
result = ner(text)

# Display results
print(result)


# Example text
text = "There is a company named EMC in Chicago."

# Apply NER to the text
result = ner(text)

# Display results
print(result)


# Example 5 -- Question Answering

question_answerer = pipeline("question-answering")

# Context and question
context = """
Hugging Face is a company that specializes in Natural Language Processing. 
They offer a platform for training, deploying, and creating models that handle various language tasks. 
One of their most notable contributions is the Transformers library, which provides state-of-the-art machine learning models.
"""

question = "What is the Transformers library known for?"

# Apply the question answering pipeline
answer = question_answerer(question=question, context=context)

# Display the answer
print(answer)