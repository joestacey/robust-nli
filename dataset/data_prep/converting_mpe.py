import pandas as pd
import nltk
import pdb
import os

def combine_premises(row):
    #import pdb; pdb.set_trace()
    '''
    Index([u'ID', u'premise1', u'premise2', u'premise3', u'premise4',
       u'hypothesis', u'entailment_judgments', u'neutral_judgments',
       u'contradiction_judgments', u'gold_label'],
      dtype='object')
    '''
    return " ".join([nltk.word_tokenize(row["premise%d"%(i)].split('/')[1]) for i in range(1,5)])

for f in ["train", "dev", "test"]:
  line_count = -1
  lbls, hypoths, premises = [], [], []

  #df = pd.read_csv("mpe/mpe_%s.txt" % (f), sep="\t")
  #df_premise =  df.iloc[:,1].str.split('/')
  
  #df['combined_premises'] = df.apply(combine_premises, axis=1)
  
  for line in open("mpe/mpe_%s.txt" % (f)):
    line_count += 1
    if line_count == 0:
      continue
    line = line.split("\t")
    assert (len(line) == 10, "MPE %s file has a bad line at line numbers %d" % (f, line_count))
    for i in range(4):
        # there are four premises per hypothesis
        lbls.append(line[-1].strip().split()[-1])
        hypoths.append(" ".join(nltk.word_tokenize(line[5].strip())))
        premises.append(" ".join(nltk.word_tokenize(line[i+1].split('/',1)[1].strip())))
    

  if f == "dev":
    f = "val"

  assert(len(hypoths) == len(set(hypoths)), "A hypothesis appears more than once")

  assert(len(lbls) == len(hypoths), "Number of labels and hypothesis for MPE %s do not match" % (f))
  lbl_out = open("mpe/labels.%s" % (f), "wb")
  source_out = open("mpe/s2.%s" % (f), "wb")
  prem_out = open("mpe/s1.%s" % (f), "wb")
  for i in range(len(lbls)):
    lbl_out.write(lbls[i].strip() + "\n")
    source_out.write(hypoths[i] + "\n")
    prem_out.write(premises[i] + "\n")
  lbl_out.close()
  source_out.close()
  prem_out.close()


os.rename("mpe/s1.val", "mpe/s1.dev")
os.rename("mpe/s2.val", "mpe/s2.dev")
os.rename("mpe/labels.val", "mpe/labels.dev")
