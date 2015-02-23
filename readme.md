Introduction: what is this?
----
This directory contains all the scripts and data files to replicate our results.
Some scripts may require you to register at Freesound.org for a free API key.

Most people are probably interested in our models. The folder `models` contains
a file called `tag-lsa.py` that contains all the code to build a tag-based distributional model. This model performs pretty well on the MEN-dataset, as you can see from the results using `test_suite.py`. We made this file as general as possible, so you can evaluate your own models using the same code. The `evaluate_model` function takes a matrix, list of row labels, and the name of the test set as its argument. It returns a dictionary with the results. The `evaluate_word2vec` function takes a word2vec model (created using Gensim), and the name of the test set as its argument. Its output is the same. For the sound models, there are two scripts inside the `models` folder: `sound_database.py` preprocesses the data and splits it into a training set and validation set. `BoAW.py` contains the code to create the BoAW-models.

###Important
* Credit the creators of the sounds if you plan on using the sounds in
any of your projects. Be aware that some sounds may have a more restrictive
license than others.
* **Please consider donating to the Freesound project.**

Required Python packages:
-----
To run all of our experiments, you will need the following packages:

 Package        | What does it do?  
----------------|------------------
 Gensim         | Topic modeling
 Networkx       | Network interface
 numpy          | Math
 python-louvain | Network analysis  
 scikit-learn   | Machine learning  
 unicodecsv     | Unicode I/O
 tabulate       | LaTeX results table
 pydub     | Sound processing toolbox
features | Sound processing toolbox