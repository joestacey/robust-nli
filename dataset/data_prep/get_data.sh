

mkdir recast
cd recast
wget https://github.com/decompositional-semantics-initiative/DNC/raw/master/inference_is_everything.zip
unzip inference_is_everything.zip
rm inference_is_everything.zip
cd ../
echo "About to split the data into formats for train.lua and eval.lua"
python converting_recast_data.py

echo "Downloading SNLI"
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip

echo "Reformatting SNLI dataset"
python converting_snli.py

#echo "Downloading GloVe"
#mkdir embds
#cd embds
#curl -LO http://nlp.stanford.edu/data/glove.840B.300d.zip
#jar xvf glove.840B.300d.zip 
#rm glove.840B.300d.zip
#cd ../ 

echo "Downloading multi-NLI"
wget http://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
unzip multinli_1.0

echo "Reformatting Multi-NLI dataset"
python converting_mnli.py

echo "Downloading MPE"
mkdir mpe
curl https://raw.githubusercontent.com/aylai/MultiPremiseEntailment/master/data/MPE/mpe_train.txt -o mpe/mpe_train.txt
curl https://raw.githubusercontent.com/aylai/MultiPremiseEntailment/master/data/MPE/mpe_dev.txt -o mpe/mpe_dev.txt
curl https://raw.githubusercontent.com/aylai/MultiPremiseEntailment/master/data/MPE/mpe_test.txt -o mpe/mpe_test.txt
python converting_mpe.py

echo "Downloading add-1 RTE"
mkdir add-one-rte
cd add-one-rte
wget http://www.seas.upenn.edu/~nlp/resources/AN-composition.tgz
tar -zxvf AN-composition.tgz 
rm AN-composition.tgz 
cd ../ 
python converting_add_1_rte.py

echo "Downloading SICK"
mkdir sick
cd sick
#wget http://clic.cimec.unitn.it/composes/materials/SICK.zip
wget https://zenodo.org/record/2787612/files/SICK.zip
unzip SICK.zip
rm SICK.zip
cd ../
python converting_sick.py

echo "Downloading SciTail"
mkdir scitail
cd scitail
wget http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.zip
unzip SciTailV1.zip
rm SciTailV1.zip
cd ../
python converting_scitail.py

echo "Downloading JOCI"
mkdir joci
cd joci
wget https://github.com/sheng-z/JOCI/raw/master/data/joci.csv.zip
unzip joci.csv.zip
cd ..
python converting_joci.py

echo "Downloading HANS"
mkdir HANS
cd HANS
wget https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt
cd ..
python converting_HANS.py

echo "Downloading SNLI-hard"
mkdir snli_hard
cd snli_hard
wget https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl
cd ..
python converting_snli_hard.py
