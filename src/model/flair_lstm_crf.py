from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


# define columns
columns = {0: 'text', 1: 'arg'}

# this is the folder in which train, test and dev files reside
data_folder = '/home/research/interns/talhindi/talhindi/data/WM/'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')
print(corpus)

tag_type = 'arg'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)


# initialize sequence tagger
tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=TransformerWordEmbeddings('bert-base-cased'),
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# initialize trainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus)


# 7. start training
trainer.train('/home/research/interns/talhindi/talhindi/models/flair/',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)



