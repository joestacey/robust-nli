import pandas as pd
import nltk
import json
import os
import shutil

for f in ["train", "dev", "test"]:
  if f != "test_hard":
    df = pd.read_table("snli_1.0/snli_1.0_%s.txt" % (f))
    if f == "dev":
      f = "val"

    sentence_ones = df['sentence1']
    sentence_twos = df['sentence2']
    labels = df['gold_label']

  assert(len(labels) == len(sentence_ones) == len(sentence_twos))
  lbl_out = open("snli_1.0/labels.%s" % (f), "wb")
  source_out = open("snli_1.0/s2.%s" % (f), "wb")
  premise_out = open("snli_1.0/s1.%s" % (f), "wb")

  label_set = set(["entailment","neutral","contradiction"])

  for i in range(len(labels)):
    if labels[i] not in label_set:
      continue
    try:
      if sentence_twos[i].isdigit() or sentence_ones[i].isdigit():
        continue
      lbl_out.write(labels[i].strip() + "\n")
      source_out.write(" ".join(nltk.word_tokenize(sentence_twos[i].strip())) + "\n")
      premise_out.write(" ".join(nltk.word_tokenize(sentence_ones[i].strip())) + "\n")
    except:
      continue
      # There are a lot of examples where only the premise sentence was given
      # The sentence is often something like: cannot see the picture

  lbl_out.close()
  source_out.close()
  premise_out.close()

os.rename("snli_1.0/s1.val", "snli_1.0/s1.dev")
os.rename("snli_1.0/s2.val", "snli_1.0/s2.dev")
os.rename("snli_1.0/labels.val", "snli_1.0/labels.dev")

#Move files to an easier folder

if not os.path.exists("SNLI/"):
  os.mkdir("SNLI")

os.rename("snli_1.0/s1.dev", "SNLI/s1.dev")
os.rename("snli_1.0/s2.dev", "SNLI/s2.dev")
os.rename("snli_1.0/labels.dev", "SNLI/labels.dev")

os.rename("snli_1.0/s1.train", "SNLI/s1.train")
os.rename("snli_1.0/s2.train", "SNLI/s2.train")
os.rename("snli_1.0/labels.train", "SNLI/labels.train")

os.rename("snli_1.0/s1.test", "SNLI/s1.test")
os.rename("snli_1.0/s2.test", "SNLI/s2.test")
os.rename("snli_1.0/labels.test", "SNLI/labels.test")

#Copying files for SNLI-hard dev set
shutil.copy2('SNLI/labels.dev','snli_hard/labels.dev')
shutil.copy2('SNLI/s1.dev','snli_hard/s1.dev')
shutil.copy2('SNLI/s2.dev','snli_hard/s2.dev')
