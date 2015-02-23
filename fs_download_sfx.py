import unicodecsv   # For I/O
import freesound    # The Freesound.org API
import time         # To pause between calls to the website.
import json

def list_to_id_filter(sfx):
    sfxfilter = ' OR '.join(map(str,sfx))
    return "id:(" + sfxfilter + ")"

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

c = freesound.FreesoundClient() # Initialize the client
c.set_token(c.client_secret)    # And set the security token

def get_pager(my_filter, my_fields):
    return c.text_search(  query='',
                            page_size=150,
                            fields=my_fields,
                            filter=my_filter
                        )

f = ['id', 'username',                                                   # ID info
    'name', 'description', 'tags', 'avg_rating', 'num_ratings',          # User-supplied descriptions
    'type', 'duration', 'bitrate', 'bitdepth', 'channels', 'samplerate', # Sound information
    'license', 'url','previews']

wanted_fields = ','.join(f)

with open("data/soundfx_ids.json") as f:
    sfx = json.load(f)

all_results  = []

for chunk in chunks(sfx,150):
    time.sleep(1)
    pager = get_pager(list_to_id_filter(chunk), wanted_fields)
    all_results += pager.results

def transform_dict(d):
    d = d.copy()
    previews = d.pop('previews')
    d['tags'] = ' '.join(d['tags'])
    d['preview-hq-ogg'] = previews['preview-hq-ogg']
    d['preview-hq-mp3'] = previews['preview-hq-mp3']
    return d

transformed_results = [transform_dict(d) for d in all_results]
names = sorted(transformed_results[0].keys())

with open('data/sfx_data.csv','w') as f:
    writer = unicodecsv.writer(f)
    writer.writerow(names)
    for result in transformed_results:
        writer.writerow([result[key] for key in names])
