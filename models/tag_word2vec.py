import community, gensim
import networkx as nx

def word2vecmodel(tagfile, threshold=5):
    with open(tagfile) as f:
        tags = [line.split()[1:] for line in f.readlines()]
    
    model = gensim.models.Word2Vec(tags, min_count=threshold)
    model.init_sims(replace=True) # reclaim memory
    return model

################################################################################
# Test functions, checking whether the model can predict similarity values.
################################################################################

import test_suite
from tabulate import tabulate

def print_table():
    table = []
    model = word2vecmodel('../data/sfx_tags.txt')
    men_result      = test_suite.evaluate_word2vec(model, measure='men')
    simlex_result   = test_suite.evaluate_word2vec(model, measure='simlex')
    table.append(['word2vec-sfx', abs(men_result['correlation']), abs(simlex_result['correlation'])])
    
    model = word2vecmodel('../data/all_tags.txt')
    men_result      = test_suite.evaluate_word2vec(model, measure='men')
    simlex_result   = test_suite.evaluate_word2vec(model, measure='simlex')
    table.append(['word2vec-all', abs(men_result['correlation']), abs(simlex_result['correlation'])])
    
    header = ['Model','MEN','SimLex-999']
    print tabulate(table, header, tablefmt="latex_booktabs")

print_table()

################################################################################
# Create graph on the basis of the model
################################################################################

def create_graph(model,n=10):
    "Create graph on the basis of the LDA model."
    
    def tuples(word,n):
        "Helper function"
        return map(
                    lambda s:tuple(sorted(s)),
                    zip([word]*n,[t[0] for t in model.most_similar(word)])
                    )
    
    G = nx.Graph()
    G.add_edges_from({p for word in model.vocab.keys()
                        for p in tuples(word,model,n)}
                    )
    return G

################################################################################
# Analyze the graph produced above.
################################################################################

# Here
# - we look at the number of isolated networks in the graph, the biggest isolated network is our object of study.
# - we make use of the wonderful python-louvain package (the community module) to partition the graph:
def graph_analysis(G):
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
    return {'subgraphs': num_graphs, 'num_clusters': num_partitions, 'modularity': mod, 'size': size}

def range_analysis(i):
    if i > 10:
        i = 10
        print "Range is limited to 10. Set i to 10."
    d = {'modularity':[], 'num_clusters': []}
    for i in range(1,i+1):
        print i
        analysis = graph_analysis(create_graph(i))
        d['modularity'].append(analysis['modularity'])
        d['num_clusters'].append(analysis['num_clusters'])
    return d

# import matplotlib.pyplot as plt
#
# d = range_analysis(10)
# y = d['modularity']
# z = d['num_clusters']
# plt.plot(range(1,11),y)
# plt.show()
