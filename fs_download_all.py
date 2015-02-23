import unicodecsv   # For I/O
import freesound    # The Freesound.org API
import time         # To pause between calls to the website.

c = freesound.FreesoundClient() # Initialize the client
c.set_token(c.client_secret)    # And set the security token

f = ['id', 'username',                                                   # ID info
    'name', 'description', 'tags', 'avg_rating', 'num_ratings',          # User-supplied descriptions
    'type', 'duration', 'bitrate', 'bitdepth', 'channels', 'samplerate', # Sound information
    'license', 'url','previews']

wanted_fields = ','.join(f)

def get_pager(my_fields):
    return c.text_search(  query='',
                            page_size=150,
                            fields=my_fields)

all_results  = []
pager        = get_pager(wanted_fields)
all_results += pager.results

print "Searching..."

while True:
    time.sleep(1) # Do NOT modify this. Unless you want it to sleep longer of course ;)
    # There isn't always a next page. If we are on the last page, pager.next_page() will throw
    # an AttributeError. The next bit of code catches that error and breaks out of the while-loop.
    try:
        pager = pager.next_page()
    except AttributeError:
        break
    print "Next page..."
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

with open('data/all_freesound_data.csv','w') as f:
    writer = unicodecsv.writer(f)
    writer.writerow(names)
    for result in transformed_results:
        writer.writerow([result[key] for key in names])

print "Done."
