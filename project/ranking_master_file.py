import regex
import re
import html
import csv
import subprocess
from time import time
from time import sleep

# our own modified version of ev.py to work inside our script for convenience
import ev

import xml.etree.ElementTree as ET

from nltk import wordpunct_tokenize

import numpy as np
from numpy.linalg import norm
import scipy.spatial.distance as ssd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

TRAINING_SET_1 = '../data/SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml'
TRAINING_SET_2 = '../data/SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml'
TEST_SET = '../data/test_input.xml'


# do you want to output a score?
# SCORE = False
SCORE = True


def scrape_vocab(xmlfile, scraped_vocab):
    '''Collect all the text in xmlfile and append it to scraped_vocab'''

    tree = ET.parse(xmlfile)
    root = tree.getroot()

    vocab = scraped_vocab

    for child in root:
        questions_sub = ""
        for item in child:
            if item.tag == 'RelQuestion':
                for subitem in item:
                    if subitem.text:
                        questions_sub += subitem.text + " "
                vocab.append(questions_sub)

            elif item.tag == 'RelComment':
                for subitem in item:
                    if subitem.text:
                        vocab.append(subitem.text)

    return vocab


def build_corpus(training_set_1, training_set_2):
    ''' build the corpus from the xml files and output the corpus as a list of questions and comments'''

    # build the corpus from the two given datasets
    vocab = scrape_vocab(training_set_1, [])
    vocab = scrape_vocab(training_set_2, vocab)

    for i in range(len(vocab)):
        # remove html character encodings
        vocab[i] = html.unescape(vocab[i])

        # lowercase corpus
        vocab[i] = vocab[i].lower()

        # tokenize corpus
        vocab[i] = wordpunct_tokenize(vocab[i])

        # remove punctuation ONLY
        vocab_i_want = []

        for token in vocab[i]:
            # check if number
            number = re.match(r'^[0-9]+$', token)
            #check if alpha
            alpha = regex.match(r'^\p{L}+$', token)
            if number or alpha:
                vocab_i_want.append(token)

        vocab[i] = ' '.join(vocab_i_want)

    return vocab


def write_corpus_to_file(corpus, outputfile):
    '''writes corpus out to outputfile'''

    # save corpus to text file ready for word2vec
    text = ' '.join(corpus)

    # avoid encoding error
    text = text.encode('ascii', 'ignore')
    text = text.decode('ascii')
    text_file = open(outputfile, "w")
    text_file.write(text)
    text_file.close()


def word2vec_embed(inputfile, outputfile, size=200, window=10, cbow=0, iterations=10):
    ''' external library word2vec is run from within the script '''

    word2vec = ['word2vec', '-train', inputfile, '-output', outputfile, '-size', str(size), '-window', str(window), '-cbow', str(cbow), '-iter', str(iterations)]

    print("Running word2vec with params: ")
    print("  size   " + str(size))
    print("  window " + str(window))
    print("  cbow   " + str(cbow))
    print("  iter   " + str(iterations))

    # word2vec is installed as a command line tool
    subprocess.call(word2vec)


def build_embeds_dictionary(inputfile):
    ''' build the embeddings into a local dictionary file to be used in the similarity measure'''

    embeds = {}

    read_input = []

    with open(inputfile, 'r') as f:
        next(f)  # skip headings
        next(f)  # skip <s/> space character embedding
        reader = csv.reader(f, delimiter=' ')
        for item in reader:
            read_input.append(item)

    for item in read_input:
        embed_vector = []
        for val in item[1:-1]:
            embed_vector.append(float(val))

        embed_vector = str(embed_vector)

        embed_vector = embed_vector[1:-1]

        embeds[item[0]] = embed_vector

    return embeds


def build_test_data_set(inputfile):
    '''using the dev set we build a data dictionary to be used in our similarity measure'''

    tree = ET.parse(inputfile)
    root = tree.getroot()

    data = {}

    for idx, child in enumerate(root):
        data[idx] = {}
        comments = []
        question_text = []
        for item in child:
            if item.tag == 'RelQuestion':
                for subitem in item:
                    try:
                        text = html.unescape(subitem.text)
                        text = text.lower()
                        tokenz = wordpunct_tokenize(text)
                        question_text.append(tokenz)
                    except:
                        pass
                data[idx]['question'] = [item for sublist in question_text for item in sublist]
                data[idx]['question_id'] = item.attrib['RELQ_ID']
            elif item.tag == 'RelComment':
                for subitem in item:
                    try:
                        text = html.unescape(subitem.text)
                        text = text.lower()
                        tokenz = wordpunct_tokenize(text)
                        comments.append([tokenz, item.attrib['RELC_ID'], item.attrib['RELC_RELEVANCE2RELQ']])
                    except:
                        print("error")
            else:
                print(item.tag)

            data[idx]['comments'] = comments

    return data


def calc_sim_embeds(input_data):
    '''similarity is run and predictions are output to outputfile'''

    for key in input_data:
        query_matrix = []
        for toke in input_data[key]['question']:
            if toke in embeds:
                temp_embed_vect = np.fromstring(embeds[toke], sep=',')
                query_matrix.append(temp_embed_vect)

        query_vector = np.sum(np.array(query_matrix), axis=0) / len(query_matrix)

        for idx, comment in enumerate(input_data[key]['comments']):
            comment_matrix = []
            for toke_comment in comment[0]:
                if toke_comment in embeds:
                    temp_embed_vect = np.fromstring(embeds[toke_comment], sep=',')
                    comment_matrix.append(temp_embed_vect)
            if len(comment_matrix) < 1:
                comment_matrix.append(np.zeros(200))

            comment_vector = np.sum(np.array(comment_matrix), axis=0) / len(comment_matrix)

            similarity = cosine_sim(query_vector, comment_vector)

            input_data[key]['comments'][idx].append(similarity)

    return input_data


def write_pred_file(input_data, output_pred_file):

    csv.register_dialect('CSV', delimiter='\t', quoting=csv.QUOTE_NONE)

    with open(output_pred_file, 'w', newline='') as f:

        writer = csv.writer(f, 'CSV')

        for i in range(len(input_data)):
            qid = input_data[i]['question_id']
            for j in input_data[i]['comments']:
                cid = j[1]
                rank = '0'
                score = j[3]
                label = 'true'

                prediction_line = [qid, cid, rank, score, label]

                writer.writerow(prediction_line)

def cosine_sim(v1, v2):
    '''self explanatory - here we take the opposite of the cosine dissimilarity'''

    if norm(v2) == 0:
        cosine_similarity = 0.5
    else:
        cosine_diff = ssd.cosine(v1, v2)
        cosine_similarity = (cosine_diff * -1) + 1
    return cosine_similarity

def score(pred_file):
    '''uses the given scorer to calculate MAP for our pred_file'''
    ev.main('SemEval2017-Task3-CQA-QL-dev-subtaskA.xml.subtaskA.relevancy', pred_file)

def lsi(corpus, data_from_test):

    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    dtm = vectorizer.fit_transform(corpus)
    vectorizer.get_feature_names()
    # LSA dimension determined empirically to be 170
    lsa = TruncatedSVD(170, algorithm='arpack')
    dtm_lsa = lsa.fit_transform(dtm)
    dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)

    for key in data_from_test:
        query_matrix = []
        for toke in data_from_test[key]['question']:
            if toke in vectorizer.vocabulary_:
                temp_embed_vect = lsa.components_[:, vectorizer.vocabulary_[toke]]
                query_matrix.append(temp_embed_vect)

        query_vector = np.sum(np.array(query_matrix), axis=0) / len(query_matrix)

        for idx, comment in enumerate(data_from_test[key]['comments']):
            comment_matrix = []
            for toke_comment in comment[0]:
                if toke_comment in vectorizer.vocabulary_:
                    temp_embed_vect = lsa.components_[:, vectorizer.vocabulary_[toke_comment]]
                    comment_matrix.append(temp_embed_vect)
            if len(comment_matrix) < 1:
                comment_matrix.append(np.zeros(100))

            comment_vector = np.sum(np.array(comment_matrix), axis=0) / len(comment_matrix)

            similarity = cosine_sim(query_vector, comment_vector)

            data_from_test[key]['comments'][idx].append(similarity)

    return data_from_test


def read_in(pred_file):

    read_input = []

    with open(pred_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for item in reader:
            read_input.append(item)

    return read_input


def combine(pred1, pred2, alpha=0.05):

    for idx, item in enumerate(pred1):
        pred1[idx][3] = str(((1-alpha)*float(item[3])) + (alpha * float(pred2[idx][3])))

    return pred1


def flatten_own_answerers(test_set, mixture):

    tree = ET.parse(test_set)
    root = tree.getroot()

    comments_to_flatten = []

    for child in root:
        questioner = None
        answerers = []
        # quality = []
        comment_id = []

        for item in child:
            if item.tag == 'RelQuestion':
                questioner = item.attrib['RELQ_USERID']
            elif item.tag == 'RelComment':
                answerers.append(item.attrib['RELC_USERID'])
                # quality.append(item.attrib['RELC_RELEVANCE2RELQ'])
                comment_id.append(item.attrib['RELC_ID'])

        if questioner in answerers:

            indices = [i for i, x in enumerate(answerers) if x == questioner]

            for item in indices:

                comments_to_flatten.append(comment_id[item])

    for idx, item in enumerate(mixture):
        if item[1] in comments_to_flatten:
            mixture[idx][3] = 0.0

    return mixture


def comment_length_enhancement(test_set, mixture, beta):
    '''Collect all the text in xmlfile and append it to the string scraped_vocab'''

    tree = ET.parse(test_set)
    root = tree.getroot()

    comments_text = []
    comments_ids = []

    for child in root:
        for item in child:

            if item.tag == 'RelComment':

                for subitem in item:
                    if subitem.text:
                        comments_text.append(subitem.text)
                        comments_ids.append(item.attrib['RELC_ID'])

    comment_lengths = [len(i) for i in comments_text]
    max_comment_length = float(max(comment_lengths))

    for idx, item in enumerate(mixture):
        if item[1] in comments_ids:
            index = comments_ids.index(item[1])
            mixture[idx][3] = str((1*float(mixture[idx][3])) + (beta * len(comments_text[index])/max_comment_length))

    return mixture


def write_out(mixture, outputfile):

    csv.register_dialect('CSV', delimiter='\t', quoting=csv.QUOTE_NONE)

    with open(outputfile, 'w', newline='') as f:
        writer = csv.writer(f, 'CSV')

        for i in range(len(mixture)):
                writer.writerow(mixture[i])


if __name__ == '__main__':

    t0 = time()

    # ===========================================================================
    # SCRAPE VOCAB FROM TRAINING DATA INTO LIST
    # ==========================================================================

    corpus = build_corpus(TRAINING_SET_1, TRAINING_SET_2)

    # ===========================================================================
    # WORD EMBEDDINGS
    # ==========================================================================

    test_data = build_test_data_set(TEST_SET)

    embeds_corpus_file = 'embeds_corpus_file.txt'
    word_embeddings_file = 'w2v.txt'
    embeddings_pred_file = 'embedding_output.txt'

    write_corpus_to_file(corpus, embeds_corpus_file)

    word2vec_embed("embeds_corpus_file.txt", word_embeddings_file)

    # sleep for 5 seconds just to make sure word_embeddings_file appears in the folder
    sleep(5)

    embeds = build_embeds_dictionary(word_embeddings_file)

    input_data = calc_sim_embeds(test_data)

    write_pred_file(input_data, embeddings_pred_file)

    if SCORE:
        score(embeddings_pred_file)

    # ===========================================================================
    # LATENT SEMANTIC INDEXING
    # ==========================================================================

    test_data = build_test_data_set(TEST_SET)

    lsi_pred_file = 'lsi_output.txt'

    output_data = lsi(corpus, test_data)

    write_pred_file(output_data, lsi_pred_file)

    if SCORE:
        score(lsi_pred_file)

    # ===========================================================================
    # COMBINE MODELS AND HAND CRAFTED FEATURES
    # ==========================================================================

    test_data = build_test_data_set(TEST_SET)

    combined_file = "mixture.txt"

    pred1 = read_in(embeddings_pred_file)

    pred2 = read_in(lsi_pred_file)

    # use these to tune alpha and beta

    # alphas = []
    # for a in range(101):
    #     alphas.append(a / 100.0)
    #
    # betas = []
    # for b in range(101):
    #     betas.append(b / 100.0)

    # ALPHA = 0 tuned with dev set
    mixture = combine(pred1, pred2, 0)

    flatten = flatten_own_answerers(TEST_SET, mixture)

    # BETA = 0.045 tuned with dev set
    final = comment_length_enhancement(TEST_SET, flatten, 0.045)

    write_out(final, combined_file)

    if SCORE:
        score(combined_file)
