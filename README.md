# Word Embeddings

`master.py` is a script that scrapes text from xml files and creates word embeddings from them, it also builds latent semantic indeces from the xml files. Following this, it uses these embeddings and LSI components to compute similarity between the question and comment and rank them appropriately.

The values are then combined in an `(1-α)(score1) + (α)(score2)`. Alpha(`α`) was tweaked on the given development set.

Following this, two enhancements are run on the combined set. If a comment has the same userid as the question answerer the ranking of the comment is set to `0`. Also a weighted subtraction is made from comments based on length.

The scorer `ev.py` that was modified and made to run within our script. This was done mainly for convenience.

The process is as follows:

1. All text is scraped from the training data including question subject, question text and comment text.

2. HTML characters in the scraped text are removed, words are lowercased, the text is tokenized and all punctutation
    subsequently filtered out.

3. Now the corpus is ready for the embedding algorithm. An third party external command line tool, word2vec developed by Google Inc., is
    used to build word embeddings from the corpus. The variables that can be set are:

+ dimension of word embeddings
+ size of window around each word
+ to use CBOW or Skip-Gram algorithms
+ the number of iterations to use

    We tuned our parameters on the development data set and set the defaults for each of these as seen in the function definition for 'word2vec_embed'

4. Once the embeddings have been formed, they are called in from an external text file into a local dictionary.

5. This dictionary is then used to calculate the cosine similarity measure between each Question and its
    10 related Comments

6. The similarity is then output by the system as a prediction file with one prediction per line.

7. This prediction is then fed into a modified version of the `ev.py` scorer script given to us for this task.

8. A similar process is run for the Latent Semantic Analysis.

9. The embeddings and semantic analysis both produce prediction files. These files are combined into a mixture file.

10. Following the creation of the mixture file as explained we squash down any comments written by the questioner to `0` and add a weighted comment length to the ranking (also modulated by a beta coefficient).

11. In the end this final modified mixture file is fed into our scorer again to assess our model.

Built by:

Micah Friedland & Robert Bevan
***

Pre-requisites:
This software was built to run on Python 3+.

The following packages need to be installed for it to run:
- regex
- re
- numpy
- scipy
- sklearn

The following software needs to be locally installed:
- word2vec

License
----

MIT
