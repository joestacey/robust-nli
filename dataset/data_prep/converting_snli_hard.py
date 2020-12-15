#
import nltk
import json_lines
labels = []
hypothesis = []
premise = []

with open('snli_hard/snli_1.0_test_hard.jsonl', 'r') as f:
  for item in json_lines.reader(f):
    labels.append(item['gold_label'])
    hypothesis.append(item['sentence2'])
    premise.append(item['sentence1'])

labels_output = open("snli_hard/labels.test", "wb")
hypothesis_output = open("snli_hard/s2.test", "wb")
premise_output = open("snli_hard/s1.test", "wb")

for i, j in enumerate(labels):
  labels_output.write(labels[i] + "\n")
  hypothesis_output.write(" ".join(nltk.word_tokenize(hypothesis[i])) + "\n")
  premise_output.write(" ".join(nltk.word_tokenize(premise[i])) + "\n")


labels_output_2 = open("snli_hard/labels.train", "wb")
hypothesis_output_2 = open("snli_hard/s2.train", "wb")
premise_output_2 = open("snli_hard/s1.train", "wb")

for i, j in enumerate(labels):
  labels_output_2.write(labels[i] + "\n")
  hypothesis_output_2.write(" ".join(nltk.word_tokenize(hypothesis[i])) + "\n")
  premise_output_2.write(" ".join(nltk.word_tokenize(premise[i])) + "\n")



                                               
