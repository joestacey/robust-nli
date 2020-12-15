import pandas as pd
import nltk
import pdb
import os


def convert_label(score, is_test):
  score = float(score)
  if is_test:
    if score <= 3:
      return "neutral"
    elif score >= 4:
      return "entailment"
    return

  if score < 3.5:
      return "neutral"
  return "entailment"

for f in ["train", "dev", "test"]:
  hyps_dict = {}
  line_count = -1
  lbls, hypoths, premise = [], [], []
  for line in open("add-one-rte/AN-composition/addone-entailment/splits/data.%s" % (f)):
    line_count += 1
    line = line.split("\t")
    assert (len(line) == 7) # "add one rte %s file has a bad line" % (f))
    lbl = convert_label(line[0], f == "test")
    if not lbl:
      continue
    lbls.append(lbl)
    prem = line[-2].replace("<b><u>", "").replace("</u></b>", "").strip()
    hyp = line[-1].replace("<b><u>", "").replace("</u></b>", "").strip()
    hypoths.append(" ".join(nltk.word_tokenize(hyp)))
    premise.append(" ".join(nltk.word_tokenize(prem)))
    if hyp not in hyps_dict:
      hyps_dict[hyp] = 0
    hyps_dict[hyp] += 1

  if f == "dev":
    f = "val"

  print ("In %s, there are %d hypothesis sentences with more than 1 context: " % (f, len([item for key,item in hyps_dict.items() if item > 1])))
  #assert(len(hypoths) == len(set(hypoths))) #, "A hypothesis appears more than once")
  assert(len(lbls) == len(hypoths)) #, "Number of labels and hypothesis for MPE %s do not match" % (f))

  lbl_out = open("add-one-rte/labels.%s"% (f), "wb")
  source_out = open("add-one-rte/s2.%s" % (f), "wb")
  premise_out = open("add-one-rte/s1.%s" % (f), "wb")
  for i in range(len(lbls)):
    lbl_out.write(lbls[i].strip() + "\n")
    source_out.write(hypoths[i] + "\n")
    premise_out.write(premise[i] + "\n")
  lbl_out.close()
  source_out.close()
  premise_out.close()


os.rename("add-one-rte/s1.val", "add-one-rte/s1.dev")
os.rename("add-one-rte/s2.val", "add-one-rte/s2.dev")
os.rename("add-one-rte/labels.val", "add-one-rte/labels.dev")

