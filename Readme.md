
# Summer 2020 internship: claim detection project
\
\
This code was tested using Anaconda's distribution of python 3.6. To install all requried packages please run:
```
pip install -r requirements.txt
```


<br><br><br>

## Feature Extraction
To extract discrete features on the Writing Mentor data (WM2020), run `extract_discrete_features_wm.py`:
```
python extract_discrete_features_wm.py --data_dir <path_to_your_wm_data>
```
\
Your `data_dir` should have a list of essays in "tsv" format where each essay have four columns as follows, with the header in the first line:
```
sentence_id	token_id	token	label
```
\
This will create a folder named `features` under `data_dir` and output all three groups of features: structural, syntactic, and lexsyn.
<br>If you want only one group of features, you should set the flag for the others to False as the default value is True for all three groups.
For example, if you want to extract only structural features you should run:
```
python extract_discrete_features_wm.py --data_dir <path_to_your_wm_data> --extract_syntactic False --extract_lexsyn False
```
If you want to extract only lexsyn features you should run:
```
python extract_discrete_features_wm.py --data_dir <path_to_your_wm_data> --extract_structural False --extract_syntactic False 
```
\
\
To extract discrete features on the SG2017 data, run ```extract_discrete_features_sg.py```:
```
python extract_discrete_features_sg.py --data_dir <path_to_your_sg_data> --train_test_split_file <path_to_your_sg_data>/train-test-split.csv
```
Your data folder should be in the same original format provided by SG2017 in this link, where it should have both .txt and .ann files for each essay
<br> https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/argument_annotated_essays_version_2/index.en.jsp
\
\
if `--train_test_split_file` is not passed as an input, the script will assume that the `train-test-split.csv` file is in the same `data_dir` folder
as in the WM feature extractor, you have a choice to turn off any of the three flags: `extract_structural`, `extract_syntactic`, `extract_lexsyn` 
\
\
To extract bert embeddings, you can run the following script (only supports data in the WM2020 format):
```
python extract_bert_features.py --data_dir <path_to_your_wm_data> --bert_model 'bert-base-cased'
```
The choice of ```--bert_model``` is optional. It has a default value of ```bert-base-cased```.
<br>This script will export a pickle named `embeddings.p` to `--data_dir/features` directory

<br><br><br>


## CRF Model Training
To train a CRF model with any combination of discrete feature groups, you can run `python run_crf.py` as follow:
```
python run_crf.py \
--train_data_dir <path_to_your_training_data> \
--test_data_dir <path_to_your_test_data> \
--features_list [['structural_position', 'structural_position_sent', 'structural_punc'], ['lexsyn_1hop']]
```
The default settings of this script assumes that the features are in a folder named `features` inside each of the `train_data_dir` and `test_data_dir`. If that is not the case for you, please set the following parameters to your corresponding feature direcotries: `--train_feature_dir`, `--test_feature_dir`.
<br>Also, it assumes the data is in the WM2020 format, if your data is in the SG2017 raw format, please set the following parameters to `True` and provide the path to the ``train-test-split.csv``:
```
--train_is_sg True
--test_is_sg True
--train_test_split_file <path_to_train-test-split.csv>
```
\
\
To train a CRF with BERT embeddings in addition to any combination of discrete feature groups, you can run `python run_crf_emb.py` as follow (this script only supports data in the four-column tsv format we have been using for WM2020 dataset):
```
python run_crf_emb.py \
--train_data_dir <path_to_your_training_data> \
--test_data_dir <path_to_your_test_data> \
--features_list [['structural_position', 'structural_position_sent', 'structural_punc'], ['lexsyn_1hop']]
```
Optional parameters ```--train_feature_dir```, ```--test_feature_dir``` work as described above. Also, the pickled embeddings files have to be in the same directory as the discrete features. The embedding file default name is ```embedding.p``` as outputed by the feature extraction script. If the names are different please provide them to their corresponding parameters:
```
--embeddings_train_file_name <name>.p
--embeddings_test_file_name <name>.p
```





