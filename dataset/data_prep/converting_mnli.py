
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import csv 
import nltk
import os
import shutil

#https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
csv.field_size_limit(sys.maxsize)

total_skipped = 0
label_set = set(["entailment","neutral","contradiction", "hidden"])
for f in ["train", "dev_matched", "dev_mismatched"]: #, "test_matched_unlabeled", "test_mismatched_unlabeled"]:
  version = "1.0"
  if "test" in f:
    version = "0.9"
  with open("multinli_1.0/multinli_%s_%s.txt" % (version, f), "rb") as tsvin:
    tsvin = csv.reader(tsvin, delimiter='\t', quoting=csv.QUOTE_NONE)
  
    lbl_out = open("multinli_1.0/labels.%s" % (f), "wb")
    source_out = open("multinli_1.0/s2.%s" % (f), "wb")
    prem_out = open("multinli_1.0/s1.%s" % (f), "wb")

    line_num = -1
    header_size = 0
    for row in tsvin:
      line_num += 1
      if line_num == 0:
       header_size = len(row)
       continue

      if len(row) != header_size:
        print "Skipping line number %d in %s" % (line_num,f)
        total_skipped += 1
        continue
      
      lbl, sent1, sent2 = row[0].strip(), row[5].strip(), row[6].strip()
      if lbl not in label_set:
        print "Skipping line number %d in %s because of label: %s" % (line_num,f, lbl)
        total_skipped += 1
        continue

      lbl_out.write(lbl.strip() + "\n")
      prem_out.write(" ".join(nltk.word_tokenize(sent1.strip())) + "\n")
      source_out.write(" ".join(nltk.word_tokenize(sent2.strip())) + "\n")
  lbl_out.close()
  source_out.close()
  prem_out.close()

print "Skipped a total of %d sentences: " % (total_skipped)

if not os.path.exists("MNLI_mismatched/"):
  os.mkdir("MNLI_mismatched")

if not os.path.exists("MNLI_matched/"):
  os.mkdir("MNLI_matched")


#Creating matched dataset
shutil.copy2('multinli_1.0/labels.dev_matched','MNLI_matched/labels.test')
shutil.copy2('multinli_1.0/labels.dev_mismatched','MNLI_matched/labels.dev')

shutil.copy2('multinli_1.0/s1.dev_matched','MNLI_matched/s1.test')
shutil.copy2('multinli_1.0/s1.dev_mismatched','MNLI_matched/s1.dev')

shutil.copy2('multinli_1.0/s2.dev_matched','MNLI_matched/s2.test')
shutil.copy2('multinli_1.0/s2.dev_mismatched','MNLI_matched/s2.dev')

shutil.copy2('multinli_1.0/s1.train','MNLI_matched/s1.train')
shutil.copy2('multinli_1.0/s2.train','MNLI_matched/s2.train')
shutil.copy2('multinli_1.0/labels.train','MNLI_matched/labels.train')

#Creating mismatched dataset
shutil.copy2('multinli_1.0/labels.dev_mismatched','MNLI_mismatched/labels.test')
shutil.copy2('multinli_1.0/labels.dev_matched','MNLI_mismatched/labels.dev')

shutil.copy2('multinli_1.0/s1.dev_mismatched','MNLI_mismatched/s1.test')
shutil.copy2('multinli_1.0/s1.dev_matched','MNLI_mismatched/s1.dev')

shutil.copy2('multinli_1.0/s2.dev_mismatched','MNLI_mismatched/s2.test')
shutil.copy2('multinli_1.0/s2.dev_matched','MNLI_mismatched/s2.dev')

shutil.copy2('multinli_1.0/s1.train','MNLI_mismatched/s1.train')
shutil.copy2('multinli_1.0/s2.train','MNLI_mismatched/s2.train')
shutil.copy2('multinli_1.0/labels.train','MNLI_mismatched/labels.train')


