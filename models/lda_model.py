import gensim
from gensim import corpora

with open('../data/all_tags.txt') as f:
    sounds = [line.split()[1:] for line in f.readlines()]

dictionary = corpora.Dictionary(sounds)
corpus = [dictionary.doc2bow(sound) for sound in sounds]


def topics(num_topics=100):
    lda = gensim.models.ldamulticore.LdaMulticore(  corpus=corpus,
                                                    id2word=dictionary,
                                                    workers=2,
                                                    num_topics=100,
                                                    chunksize=10000,
                                                    passes=1)
    return [[i.split('*')[1] for i in x.split(' + ')]
                              for x in lda.show_topics(num_topics)]
