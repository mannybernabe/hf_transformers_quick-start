
# 2 More Verbose

from transformers import pipeline
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification



model_name = "distilbert-base-uncased-finetuned-sst-2-english" #default model use for the pipeline
model      = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)



classifier = pipeline("sentiment-analysis",
                      model=model,
                      tokenizer=tokenizer)

result = classifier("I've been waiting for a hugging face course!")


print(result)




# Look at tokenizer

sequence = "Using a transformer network is simple"
result = tokenizer(sequence)
print(result)

tokens = tokenizer.tokenize(sequence)
print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

decode_string = tokenizer.decode(ids)
print(decode_string)



#3 combine with pytorch or tensorflow

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model      = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer  = AutoTokenizer.from_pretrained(model_name)   

classifier = pipeline("sentiment-analysis",
                      model=model,
                      tokenizer=tokenizer)

X_train = ["I've been waiting for a hugging face course my whole life.",
           "Python is the best programming language!",
           "I hate R!"
           ]

result = classifier(X_train)
print(result)

#3.2 do it separately

batch = tokenizer(X_train, 
                  padding=True, 
                  truncation=True, 
                  return_tensors="pt")

print(batch)

with torch.no_grad():
    outputs = model(**batch)
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)
    
    
# save toenizer and model to local directory

save_directory = "./model_save"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

tok=AutoTokenizer.from_pretrained(save_directory)
model=AutoModelForSequenceClassification.from_pretrained(save_directory)




    







