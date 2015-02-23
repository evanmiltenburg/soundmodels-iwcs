import unicodecsv, tabulate
from collections import Counter

def avg(l):
    "Standard average function."
    return sum(l)/float(len(l))

def print_stats(filename):
    with open(filename) as f:
        tags        = []
        duration    = []
        for row in unicodecsv.DictReader(f):
            tags.append(row['tags'].lower().split())
            duration.append(float(row['duration']))
    
    totaltags   = len([t for l in tags for t in l])
    c           = Counter()
    
    for l in tags:
        c.update(l)
    
    c2 = c.copy()
    for k in c2.keys():
        if k.isdigit() or c2[k] <= 5:
            c2.pop(k)
    
    print
    
    table = [
    ["Total number of sounds:", len(tags)                       ],
    ["Minimum duration:", min(duration)                         ],
    ["Maximum duration:", max(duration)                         ],
    ["Average duration:",    avg(duration)                      ],
    ["Total number of tags:", totaltags                         ],
    ["Average # tags per sound:", sum(c.values())/float(len(tags))],
    # len(tags) = number of tagsets = number of sounds
    ["Number of sounds per tag:", avg(c.values())               ],
    ["Number of different tags:", len(c.keys())                 ],
    ["Most common:", ' '.join(map(str,c.most_common(1)[0]))     ],
    ["Below cutoff:", len(c.keys()) - len(c2.keys())            ],
    ["Average # tags per sound (cutoff):", sum(c2.values())/float(len(tags))],
    ["Number of sounds per tag (cutoff):", avg(c2.values())               ],
    ["Number of different tags (cutoff):", len(c2.keys())                 ],
    ["Most common (cutoff):", ' '.join(map(str,c2.most_common(1)[0]))     ]]
    
    print tabulate.tabulate(table, headers = [filename, 'values']), '\n'

for f in ['data/all_freesound_data.csv','data/sfx_data.csv']:
    print_stats(f)
