import urllib2, time

with open('data/sfx_mp3.txt') as f:
    
for id,url in [line.split() for line in f.readlines()]:
    time.sleep(1)
    f = urllib2.urlopen(url)
    with open('sounds/'+id+'.mp3', "wb") as sound_file:
        sound_file.write(f.read())
