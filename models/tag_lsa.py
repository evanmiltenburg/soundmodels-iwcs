import community
import csv
import numpy as np
import networkx as nx
from scipy.stats.stats import pearsonr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from itertools import izip
import test_suite # own module to test the performance of the model.

################################################################################
# Load the tag data:
################################################################################

def load_data(filename,cutoff=5,nodigits=True):
    "Load list of tagsets with tags occurring more than the cutoff value."
    
    # open the file and get a list of tagsets:
    with open(filename) as f:
        tags    = [line.split()[1:] for line in f.readlines()]
    
    # create a counter to determine the common tags:
    c           = Counter([t for l in tags for t in l])
    
    # remove digits:
    if nodigits:
        digits  = [tag for tag in c if tag.isdigit()]
        for digit in digits:
            c.pop(digit)
    
    # get a list of the common tags:
    common_tags = set([tag for tag in c if c[tag] > cutoff])
    
    # define an internal function to select common tags:
    def common_string(t):
        "Remove infrequent tags from the tag string."
        return ' '.join(list(set(t) & common_tags))
    
    # return a list of tagsets, formatted as tag-separated strings:
    return [common_string(t) for t in tags]

def load_dm_matrix(filename,delim=' '):
    "Loads a matrix in the .dm format."
    with open(filename) as f:
        rows = [row for row in csv.reader(f,delimiter=delim)]
    tag_list = [row[0] for row in rows]
    matrix   = np.matrix([row[1:] for row in rows])
    return matrix,tag_list

################################################################################
# Generate term-term matrix, tfidf-transform the data:
################################################################################

def termterm(tags):
    "Create a term*term matrix, and perform TFIDF. Returns matrix + row labels."
    vec      = CountVectorizer(tokenizer=lambda s:s.split())         # create CountVectorizer to build a Document-Term matrix.
    data     = vec.fit_transform(tags)   # create sparse representation
    tag_list = vec.get_feature_names()   # get list of tags
    termterm = data.transpose() * data   # compute term*term matrix
    vect     = TfidfTransformer()        # create TfidfTransformer
    tfidf    = vect.fit_transform(termterm) # ..and fit it to the term-term matrix
    return (tfidf,tag_list)

################################################################################
# Concatenate the matrix generated above with some other matrix:
################################################################################

def scale_by_max(x,use_log=False):
    """Scale the matrix by dividing by the largest value inside the matrix.
    Optionally the values are first log-transformed so that very large max-values
    do not influence the outcome as much."""
    x = x.astype(float)
    x = x.view(type=np.ndarray)
    if use_log and x.max() > 100000:
        print "using log!"
        x = np.where(x < 1,0,np.log(x))
    return x/float(x.max())

def concatenate(a,a_labels,b,b_labels,normalize=False,use_log=False):
    "Concatenates two matrices, and matches the rows to each other using the row labels."
    # If the length is not the same, we cannot concatenate the two.
    if not a.shape[0] == b.shape[0] == len(a_labels) == len(b_labels):
        return "Length mismatch!"
    
    # Turn matrix A from a sparse representation into a numpy array:
    # Catch AttributeError if A is not a sparse matrix.
    try:
        a = a.toarray()
    except AttributeError:
        pass
    
    # ...and do the same for matrix B.
    try:
        b = b.toarray()
    except AttributeError:
        pass
    
    # If A and B should be on the same scale from 0 to 1:
    if normalize:
        a = scale_by_max(a,use_log)
        b = scale_by_max(b,use_log)
    
    # If the matrices are already ordered the same way, concatenate.
    if a_labels == b_labels:
        return np.concatenate((a,b),axis=1)
    
    # Otherwise, create a mapping between the two and reorder the second matrix.
    a_mapping = {label:num for num,label in enumerate(a_labels)}
    new_matrix = np.empty(b.shape)
    for num, label in enumerate(b_labels):
        new_matrix[a_mapping[label]] = b[num]
    return np.concatenate((a,new_matrix),axis=1)

def load_and_combine(mat_a, tl_a, filename, normalize=False,use_log=False):
    "Loads a .dm matrix and combines it with an existing matrix."
    mat_b, tl_b = load_dm_matrix(filename)
    return concatenate(mat_a, tl_a, mat_b, tl_b, normalize,use_log)

def sfx_combine(filename, normalize=False, use_log=False):
    tags            = load_data('../data/sfx_tags.txt')
    matrix,tag_list = termterm(tags)
    new_matrix      = load_and_combine(matrix,tag_list,filename,normalize,use_log)
    return new_matrix, tag_list

################################################################################
# Reduce dimensionality and create dictionary with most similar tags:
################################################################################

def reduce_matrix(matrix,dim=300):
    "Perform SVD with the given number of dimensions. Returns reduced matrix."
    svd     = TruncatedSVD(n_components=dim)
    return svd.fit_transform(matrix)

def create_similarity_dict(reduced,tag_list):
    """Create a dictionary containing an ordered list of similar tags for each tag.
    
    Useful for smaller matrices, but runs out of memory on larger ones."""
    d        = pairwise_distances(reduced,metric='cosine')
    return {name: [t[1] for t in sorted(zip(d[i],tag_list)) if not t[1]==name]
            for i,name in enumerate(tag_list)}

from multiprocessing.dummy import Pool as ThreadPool
from datetime import datetime
from collections import defaultdict

def get_simdict(reduced,tag_list):
    "Get dictionary of similarity values. Less memory intensive, but also less efficient."
    # Some helper functions to keep everything readable:
    cosine = lambda x,y:float(pairwise_distances(x,y,metric='cosine'))
    
    def simlist_helper(t):
        "Compute cosine similarity between two rows."
        row_a,row_b,a,b = t
        return (cosine(row_a,row_b),(a,b))
    
    def gen_tuples(tag_list):
        "Generator that ensures that we won't compute the same pair twice."
        for i,a in enumerate(tag_list):
            for j,b in enumerate(tag_list):
                if i < j:
                    yield (i,a,j,b)
    
    # Print start time:
    print str(datetime.now())
    
    # Use multiple threads:
    pool = ThreadPool(4)
    l = pool.map(simlist_helper,
                # generator expression:
                ((reduced[i],reduced[j],a,b) for i,a,j,b in gen_tuples(tag_list))
                )
    pool.close()
    pool.join()
    print str(datetime.now())
    # Create the dictionary:
    d = defaultdict(list)
    for v,(a,b) in l:
        d[a].append((v,b))
        d[b].append((v,a))
    return {k:sorted(d[k]) for k in d}


################################################################################
# Test functions, checking whether the model can predict similarity values.
################################################################################

def test_reduction(matrix,tag_list,range_of_dims=xrange(50,1050,50),test='men',verbose=False):
    "Check which dimensionality provides the best results with a given test set."
    l = []
    for i in range_of_dims:
        if verbose:
            print str(i) + '...'
        reduced = reduce_matrix(matrix,i)
        result  = test_suite.evaluate_model(reduced,tag_list,measure=test)
        l.append((abs(result['correlation']),i))
    return sorted(l,reverse=True)

# This seems to be the best for the MEN results:
#  100/400 dims for sfx.
# 3000 dims for the entire database.

# This code tests the combined models:
# matrix, tag_list = sfx_combine("model2ppmi.dm")
# test_reduction(matrix,tag_list,test='simlex')
# test_reduction(matrix,tag_list,test='men')
#
# matrix, tag_list = sfx_combine("model3plmi.dm")
# test_reduction(matrix,tag_list,test='simlex')
# test_reduction(matrix,tag_list,test='men')

from tabulate import tabulate

def print_table():
    table = []
    tags = load_data('../data/sfx_tags.txt')
    matrix,tag_list = termterm(tags)
    reduced         = reduce_matrix(matrix,100)
    men_result      = test_suite.evaluate_model(reduced,tag_list,measure='men')
    simlex_result   = test_suite.evaluate_model(reduced,tag_list,measure='simlex')
    table.append(['SoundFX-tags', 100, abs(men_result['correlation']), abs(simlex_result['correlation'])])
    
    reduced         = reduce_matrix(matrix,400)
    men_result      = test_suite.evaluate_model(reduced,tag_list,measure='men')
    simlex_result   = test_suite.evaluate_model(reduced,tag_list,measure='simlex')
    table.append(['SoundFX-tags', 400, abs(men_result['correlation']), abs(simlex_result['correlation'])])
    
    tags = load_data('../data/all_tags.txt')
    matrix,tag_list = termterm(tags)
    reduced         = reduce_matrix(matrix,3000)
    men_result      = test_suite.evaluate_model(reduced,tag_list,measure='men')
    simlex_result   = test_suite.evaluate_model(reduced,tag_list,measure='simlex')
    table.append(['Freesound-tags', 3000, abs(men_result['correlation']), abs(simlex_result['correlation'])])
                          
    matrix,tag_list = load_dm_matrix("model3plmi_svd60.dm",delim='\t')
    men_result      = test_suite.evaluate_model(matrix,tag_list,measure='men')
    simlex_result   = test_suite.evaluate_model(matrix,tag_list,measure='simlex')
    table.append(['SoundFX-BoAW', 60, abs(men_result['correlation']), abs(simlex_result['correlation'])])
    
    matrix, tag_list = sfx_combine("model3plmi.dm")
    reduced          = reduce_matrix(matrix,1000)
    men_result       = test_suite.evaluate_model(reduced,tag_list,measure='men')
    simlex_result    = test_suite.evaluate_model(reduced,tag_list,measure='simlex')
    table.append(['Combined', 1000, abs(men_result['correlation']), abs(simlex_result['correlation'])])
    
    header = ['Model','dimensions','MEN','SimLex-999']
    print tabulate(table, header, tablefmt="latex_booktabs")

def compare_tag_dm(tag_file='../data/sfx_tags.txt', dim=400,
                   dm_file="model3plmi_svd60.dm", delim='\t'):
    "Compare the best tag model with the best audio word model."
    tags = load_data(tag_file)
    matrix,tl1 = termterm(tags)
    m1 = reduce_matrix(matrix,dim)
    del tags
    del matrix
    m2,tl2 = load_dm_matrix(dm_file,delim)
    return test_suite.compare_models(m1,tl1,m2,tl2)

def check_normalization(use_log=False):
    """Check whether normalization produces better results.
    (Thanks to the reviewers for suggesting this)"""
    matrix, tag_list = sfx_combine("model3plmi.dm",normalize=True,use_log=use_log)
    reduced          = reduce_matrix(matrix,1000)
    men_result       = test_suite.evaluate_model(reduced,tag_list,measure='men')
    simlex_result    = test_suite.evaluate_model(reduced,tag_list,measure='simlex')
    print "MEN"
    print test_reduction(reduced,tag_list,range_of_dims=xrange(50,1000,50),test='men')
    print "Simlex"
    print test_reduction(reduced,tag_list,range_of_dims=xrange(50,1000,50),test='simlex')

print "Testing concatenation. First /max, then log()/max"
#check_normalization(use_log=False)
check_normalization(use_log=True)

################################################################################
# Create graph from similarity dictionary:
################################################################################

def create_edges(sim_dict,n=5):
    "Create a set of edges on the basis of a similarity dictionary."
    return {tuple(sorted(p)) for k in sim_dict for p in zip([k]*n, sim_dict[k])}

def create_graph(sd,n=5):
    "Create a graph on the basis of a similarity dictionary."
    G = nx.Graph()
    G.add_edges_from(create_edges(sd,n))
    return G

def create_save_graph(tag_file='../data/sfx_tags.txt',
                      dims=400,
                      n=5,
                      cutoff=5,
                      filename='sfx_tags.gexf'):
    "Load tags file, create model, build graph, and save for use in Gephi."
    tags = load_data(tag_file,cutoff=cutoff)
    matrix,tag_list = termterm(tags)
    reduced         = reduce_matrix(matrix,dims)
    sd  = create_similarity_dict(reduced, tag_list)
    G   = create_graph(sd,n)
    nx.write_gexf(G,'graphs/'+filename)

def write_dm_graph(dm_file='model3plmi_svd60.dm',delim='\t',
                   n=5,
                   filename='SoundFX-BoAW.gexf'):
    "Open .dm file, create graph, and save it for use in Gephi."
    m,tl = load_dm_matrix(dm_file,delim=delim)
    sd = create_similarity_dict(m,tl)
    G = create_graph(sd,n)
    nx.write_gexf(G,'graphs/'+filename)

################################################################################
# Analyze the graph produced above.
################################################################################

# Here
# - we look at the number of isolated networks in the graph, the biggest isolated network is our object of study.
# - we make use of the wonderful python-louvain package (the community module) to partition the graph:
def graph_analysis(G):
    "Analyze the graph. Returns a dictionary with useful data."
    graph_list = list(nx.connected_component_subgraphs(G))
    # Number of isolated networks:
    num_graphs = len(graph_list)
    main_graph = graph_list[0]
    for sub_graph in graph_list:
        if len(sub_graph.nodes()) > len(main_graph.nodes()):
            main_graph = sub_graph
    # size of the graph:
    size = len(main_graph.nodes())
    # do a partition analysis on the main graph.
    partition = community.best_partition(main_graph)
    # Number of partitions:
    num_partitions = max(partition.values())
    # Modularity:
    mod = community.modularity(partition,main_graph)
    return {   'subgraphs': num_graphs,
                'num_clusters': num_partitions,
                'modularity': mod,
                'size': size,
                'partition': partition  }

def graph_results_from_dm(filename,delim='\t'):
    "Loads the reduced .dm file and produces the results"
    m,tl = load_dm_matrix(filename,delim=delim)
    sd = create_similarity_dict(m,tl)
    G = create_graph(sd)
    return graph_analysis(G)

# # write big graph to file:
# tags = load_data('../data/sfx_tags.txt',cutoff=5)
# matrix,tag_list = termterm(tags)
# reduced         = reduce_matrix(matrix,100)
# sd  = create_similarity_dict(reduced, tag_list)
# G   = create_graph(sd,5)
# nx.write_gexf(G,'graphs/sfx_tags_100_5_cutoff_5.gexf')
# create_save_graph(cutoff=0)
