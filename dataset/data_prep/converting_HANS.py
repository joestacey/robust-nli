import nltk
import shutil

labels = []
hypothesis = []
premise = []

for line in open("HANS/heuristics_evaluation_set.txt"):
  line = line.split("\t")
  labels.append(line[0])
  hypothesis.append(line[6])
  premise.append(line[5])

labels = ["neutral" if x == "non-entailment" else x for x in labels]
#labels = ["neutral" if x == "n" else x for x in labels]

labels_out = open("HANS/labels.test","wb")
hypothesis_out = open("HANS/s2.test","wb")
premise_out = open("HANS/s1.test","wb")

labels_outdev = open("HANS/labels.dev","wb")
hypothesis_outdev = open("HANS/s2.dev","wb")
premise_outdev = open("HANS/s1.dev","wb")

switch=1
for i, j in enumerate(labels):
  if i > 0:
    if switch == 1:
      labels_out.write(labels[i] + "\n")
      hypothesis_out.write(" ".join(nltk.word_tokenize(hypothesis[i])) + "\n")
      premise_out.write(" ".join(nltk.word_tokenize(premise[i])) + "\n")
    if switch == -1:
      labels_outdev.write(labels[i] + "\n")
      hypothesis_outdev.write(" ".join(nltk.word_tokenize(hypothesis[i])) + "\n")
      premise_outdev.write(" ".join(nltk.word_tokenize(premise[i])) + "\n")
    switch = switch * -1

shutil.copy2('HANS/labels.dev','HANS/labels.train')
shutil.copy2('HANS/s1.dev','HANS/s1.train')
shutil.copy2('HANS/s2.dev','HANS/s2.train')
