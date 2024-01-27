
from transformers import pipeline
import torch

# Example 1 -- Sentiment analysis

classifier = pipeline("sentiment-analysis")

result = classifier("I've been waiting for a hugging face course!")

print(result)




# Example 2 -- Text generation

generator = pipeline("text-generation", model="distilgpt2")

res = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

print(res)



# Example 3 -- Zero shot classification
classifier = pipeline("zero-shot-classification")

res = classifier(
    "This is a course about Python list comprehension",
    candidate_labels=["education", "politics", "business"],
)

print(res)


