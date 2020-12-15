import pandas as pd
import nltk
import os

test = {'lbls': [], 'hypoths': [], 'prems': []}
train = {'lbls': [], 'hypoths': [], 'prems': []}
dev = {'lbls': [], 'hypoths': [], 'prems': []}
DATA = {"TEST": test, "TRAIN" : train, "TRIAL": dev}

tag_set = set()

line_count = -1
for line in open("sick/SICK.txt"):
  line_count += 1
  if line_count == 0:
    continue
  line = line.split("\t")
  tag = line[-1].strip()
  if tag not in DATA.keys():
    print "Bad tag: %s" % (tag)
  hyp = line[2]
  prem = line[1]
  #hyp = " ".join(nltk.word_tokenize(line[0].strip())) +  "|||" + " ".join(nltk.word_tokenize(line[2].strip()))
  lbl = line[3].lower()
  DATA[tag]['lbls'].append(lbl)
  DATA[tag]['hypoths'].append(hyp)
  DATA[tag]['prems'].append(prem)
  tag_set.add(line[-1])


print tag_set

for pair in [("TEST", "test"), ("TRAIN", "train"), ("TRIAL", "val")]:
  print "Number of %s examples: %d" % (pair[1], len(DATA[pair[0]]['lbls']))
  lbl_out = open("sick/labels.%s" % (pair[1]), "wb")
  source_out = open("sick/s2.%s" % (pair[1]), "wb")
  prem_out = open("sick/s1.%s" % (pair[1]), "wb")
  for i in range(len(DATA[pair[0]]['lbls'])):
    lbl_out.write(DATA[pair[0]]['lbls'][i].strip() + "\n")
    source_out.write(DATA[pair[0]]['hypoths'][i].strip() + "\n")
    prem_out.write(DATA[pair[0]]['prems'][i].strip() + "\n")
  source_out.close()
  lbl_out.close()

os.rename("sick/s1.val", "sick/s1.dev")
os.rename("sick/s2.val", "sick/s2.dev")
os.rename("sick/labels.val", "sick/labels.dev")

