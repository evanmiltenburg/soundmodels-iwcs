import unicodecsv

with open('data/sfx_data.csv') as f:
    pairs = [(row['id'],row['tags'].lower()) for row in unicodecsv.DictReader(f)]

with open('data/sfx_tags.txt','w') as f:
    for id,tags in pairs:
        line = id + ' ' + tags + '\n'
        f.write(line.encode('utf8'))

with open('data/sfx_data.csv') as f:
    pairs = [(row['id'],row['description'].replace('\n', ' ').replace('\r', ' ').lower()) for row in unicodecsv.DictReader(f)]

with open('data/sfx_descriptions.txt','w') as f:
    for id,desc in pairs:
        line = id + ' ' + desc + '\n'
        f.write(line.encode('utf8'))

with open('data/sfx_data.csv') as f:
    pairs = [(row['id'],row['preview-hq-mp3'].lower()) for row in unicodecsv.DictReader(f)]

with open('data/sfx_mp3.txt','w') as f:
    for id,url in pairs:
        line = id + ' ' + url + '\n'
        f.write(line.encode('utf8'))
