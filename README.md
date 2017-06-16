# Origins #

This repository builds on the work of Serban (see the original repository https://github.com/julianser/hed-dlg-truncated), an End-To-End Dialogue System utilizing
deep recurrent neural networks to model Natural Language Understanding (NLG), Dialogue Manager, and Natural Language Understanding (NLU) in one large model.

### Features ###

The purpose of this contribution is to provide scripts that support and automatize the model-creation, its training, and the utilization of the model beyond the original purpose of
generating answers to a given context.

The features are:

* A command line interface that allows to create models with a representative folder structure, hosting training data, configuration files, word embeddings, different training versions, and other model related data.

* Functionality to create and access a database of utterance and dialogue embeddings. The database is using the hdf5 standard (see h5py), a data format that is designed for flexible and efficient I/O and for high volume and complex data. **(not in the code yet)**

* An Approximate Nearest Neighbor (ANN) model that allows to search for similar conversations in a database of dialogue embeddings. **(not in the code yet)**

* A hybrid version of the Hierarchical Recurrent Encoder Decoder (HRED) architecture and a retrieval-based approach, the ANN model, that allows to answer questions using the database instead of generating an answer end-to-end. **(not in the code yet)**

### Setup ###

TODO


