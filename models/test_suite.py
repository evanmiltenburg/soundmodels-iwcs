import csv
from sklearn.metrics.pairwise   import pairwise_distances
from scipy.stats.stats          import spearmanr
from itertools                  import combinations

################################################################################
# Load the data:
################################################################################

def load_men():
    "loads the MEN dataset"
    with  open('../data/MEN/MEN_dataset_natural_form_full') as f:
        tuples  = {tuple(line.strip().split()) for line in f.readlines()}
        return {t[:2]:float(t[2]) for t in tuples}

def load_radinsky():
    "loads the similarity dataset by Radinsky et al."
    #See http://tx.technion.ac.il/~kirar/Datasets.html
    with open('../data/Mtruk.csv') as f:
        reader = csv.DictReader(f,fieldnames=['word1','word2','score'])
        return {(d['word1'],d['word2']):float(d['score']) for d in reader}

def load_rarewords():
    "loads the rare words dataset."
    with open('../data/rw/rw.txt') as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames=['word1','word2','score'])
        return {(d['word1'],d['word2']):float(d['score']) for d in reader}

def load_simlex():
    "loads the simlex999 dataset"
    with open("../data/SimLex-999/SimLex-999.txt") as f:
        reader  = csv.DictReader(f,delimiter='\t')
        return {(d['word1'],d['word2']):float(d['SimLex999']) for d in reader}

def load_wordsim():
    "loads the wordsim353 dataset"
    with open('../data/wordsim353/combined.csv') as f:
        reader = csv.DictReader(f)
        return {(d['Word 1'],d['Word 2']):float(d['Human (mean)']) for d in reader}

def load_wordsim_rel():
    "loads the wordsim relatedness subset."
    with open('../data/wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt') as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames=['word1','word2','score'])
        return {(d['word1'],d['word2']):float(d['score']) for d in reader}

def load_wordsim_sim():
    "loads the wordsim similarity subset."
    with open('../data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt') as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames=['word1','word2','score'])
        return {(d['word1'],d['word2']):float(d['score']) for d in reader}

# Private variable holding all the similarity data:
try:
    __resource__ = {'men':          load_men(),
                    'radinsky':     load_radinsky(),
                    'rarewords':    load_rarewords(),
                    'simlex':       load_simlex(),
                    'wordsim':      load_wordsim(),
                    'wordsim_rel':  load_wordsim_rel(),
                    'wordsim_sim':  load_wordsim_sim()}
except:
    __resource__ = {'':''}
    print("Problem loading similarity data.")

################################################################################
# The main function:
################################################################################

def evaluate_model(matrix, feature_names, measure='men'):
    "Evaluates a model on the basis of the provided similarity measure."
    # Load the similarity measure, if possible.
    try:
        sim_dict = __resource__[measure]
    except KeyError:
        return None
    
    # Select pairs that can be used for testing.
    sim_words        = {word for pair in sim_dict for word in pair}
    usable_words     = set(feature_names) & sim_words
    usable_pairs     = {key for key in sim_dict.keys()
                        if set(key).issubset(usable_words)}
    
    # Gather lists of actual values and 'predictions'
    actual_values    = []
    predicted_values = []
    indices          = {name:i for i,name in enumerate(feature_names)}
    cosine           = lambda x,y:float(pairwise_distances(x,y,metric='cosine'))
    for a,b in usable_pairs:
        actual_values.append(sim_dict[(a,b)])
        predicted_values.append(cosine(matrix[indices[a]],matrix[indices[b]]))
    
    # Compute the correlation:
    correlation, sig = spearmanr(actual_values, predicted_values)
    return {   "correlation":  correlation,
                "explained":    correlation*correlation,
                "significance": sig,
                "test_pairs":   len(usable_pairs),
                "predictions":  dict(zip(usable_pairs,predicted_values))}

def compare_models(m1,tl1,m2,tl2):
    "Test how well the two models correlate"
    # Ensure overlap between the two tag lists:
    overlap     = set(tl1) & set(tl2)
    # Get the row indices for the tags:
    m1_indices  = {name:i for i,name in enumerate(tl1)}
    m2_indices  = {name:i for i,name in enumerate(tl2)}
    # Prepare lists to collect data:
    m1_values   = []
    m2_values   = []
    differences = []
    # Create shorthand for the cosine similarity function:
    cosine      = lambda x,y:float(pairwise_distances(x,y,metric='cosine'))
    # For all combinations of tags, compute the distances
    for a,b in combinations(overlap,2):
        m1_pred = cosine(m1[m1_indices[a]],m1[m1_indices[b]])
        m2_pred = cosine(m2[m2_indices[a]],m2[m2_indices[b]])
        m1_values.append(m1_pred)
        m2_values.append(m2_pred)
        differences.append((abs(m1_pred-m2_pred), a+' '+b))
    # Correlate the two sets of distances
    correlation, sig = spearmanr(m1_values, m2_values)
    return {   "correlation":  correlation,
                "significance": sig,
                "differences":  sorted(differences,reverse=True)}

def evaluate_word2vec(model, measure='men'):
    "Evaluates a model on the basis of the provided similarity measure."
    # Load the similarity measure, if possible.
    try:
        sim_dict = __resource__[measure]
    except KeyError:
        return None
    
    # Select pairs that can be used for testing.
    sim_words        = {word for pair in sim_dict for word in pair}
    usable_words     = set(model.vocab.keys()) & sim_words
    usable_pairs     = {key for key in sim_dict.keys()
                        if set(key).issubset(usable_words)}
    
    # Gather lists of actual values and 'predictions'
    actual_values    = []
    predicted_values = []
    for a,b in usable_pairs:
        actual_values.append(sim_dict[(a,b)])
        predicted_values.append(model.similarity(a,b))
    
    # Compute the correlation:
    correlation, sig = spearmanr(actual_values, predicted_values)
    return {   "correlation":  correlation,
                "explained":    correlation*correlation,
                "significance": sig,
                "test_pairs":   len(usable_pairs),
                "predictions":  dict(zip(usable_pairs,predicted_values))}


# There is still quite a bit of overlap between the two functions.
# Maybe there is a way to simplify this, but if there aren't any other models
# then this should be fine.
