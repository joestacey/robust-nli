#_data.txt
import glob
import nltk
import os

cwd = os.getcwd()

if not os.path.exists(cwd + "/recast/dpr/"):
  os.mkdir(cwd + "/recast/fnplus/")

if not os.path.exists(cwd + "/recast/sprl/"):
  os.mkdir(cwd + "/recast/sprl/")

if not os.path.exists(cwd + "/recast/dpr/"):
  os.mkdir(cwd + "/recast/dpr/")

def main():
  train_count = 0
  val_count = 0
  test_count = 0
  input_files = glob.glob("./recast/*_data.txt")

  f_train_orig_data = open("recast/cl_train_orig_dataset_file", "wb")
  f_val_orig_data = open("recast/cl_val_orig_dataset_file", "wb")
  f_test_orig_data = open("recast/cl_test_orig_dataset_file", "wb")
  for file in input_files:

    f_type = file.split("/")[-1].split("_")[0] 
    f_train_lbl = open("recast/"+f_type+"/labels.train", "wb")
    f_dev_lbl = open("recast/"+f_type+"/labels.dev", "wb")
    f_test_lbl = open("recast/"+f_type+"/labels.test", "wb")

    f_train_hyp = open("recast/"+f_type+"/s2.train", "wb")
    f_dev_hyp = open("recast/"+f_type+"/s2.dev", "wb")
    f_test_hyp = open("recast/"+f_type+"/s2.test", "wb")


    f_train_prem = open("recast/"+f_type+"/s1.train", "wb")
    f_dev_prem = open("recast/"+f_type+"/s1.dev", "wb")
    f_test_prem = open("recast/"+f_type+"/s1.test", "wb")

    out_files = {"train": [f_train_lbl, f_train_hyp, f_train_prem], \
                "dev": [f_dev_lbl, f_dev_hyp, f_dev_prem], \
               "test": [f_test_lbl, f_test_hyp, f_test_prem]}


    orig_sent, hyp_sent, data_split, src, label = None, None, None, None, None
    for line in open(file):
      if line.startswith("entailed: "):
        label = "entailment"
        if "not-entailed" in line:
          label = "neutral"
      elif line.startswith("text: "):
        orig_sent = " ".join(line.split("text: ")[1:]).strip()
      elif line.startswith("hypothesis: "):
        hyp_sent = " ".join(line.split("hypothesis: ")[1:]).strip()
      elif line.startswith("partof: "):
        data_split = line.split("partof: ")[-1].strip()
      elif line.startswith("provenance: "):
        src = line.split("provenance: ")[-1].strip()
      elif not line.strip():
        assert orig_sent != None
        assert hyp_sent != None
        assert data_split != None
        assert src != None
        assert label != None
        '''if data_split == 'train':  and train_count > 1000:
          continue
        elif data_split == 'train':
          train_count += 1
        elif data_split == 'dev' and val_count > 1000:
          continue
        elif data_split == 'dev':
          val_count += 1 
        elif data_split == 'test' and test_count > 100:
          continue
        elif data_split == 'test':
          test_count += 1
        ''' 
        #print orig_sent, hyp_sent, data_split, src, label
        out_files[data_split][0].write(str(label) +  "\n")
        out_files[data_split][2].write(orig_sent + "\n")   
        out_files[data_split][1].write(hyp_sent + "\n")
    
        orig_sent, hyp_sent, data_split, src, label = None, None, None, None, None

  for data_type in out_files:
    out_files[data_type][0].close()
    out_files[data_type][1].close()
    out_files[data_type][2].close()

if __name__ == '__main__':
  main()

#os.rename("recast/fnplus/s1.val", "recast/fnplus/s1.dev")
#os.rename("recast/fnplus/s2.val", "recast/fnplus/s2.dev")
#os.rename("recast/fnplus/labels.val", "recast/fnplus/labels.dev")

#os.rename("recast/dpr/s1.val", "recast/dpr/s1.dev")
#os.rename("recast/dpr/s2.val", "recast/dpr/s2.dev")
#os.rename("recast/dpr/labels.val", "recast/dpr/labels.dev")

#os.rename("recast/sprl/s1.val", "recast/sprl/s1.dev")
#os.rename("recast/sprl/s2.val", "recast/sprl/s2.dev")
#os.rename("recast/sprl/labels.val", "recast/sprl/labels.dev")




