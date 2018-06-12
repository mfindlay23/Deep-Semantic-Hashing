import os

path = '/mnt/c/Users/Kevin/Desktop/Deep-Semantic-Hashing/20news-bydate-train'

total_files = 0
file_output = open('20news.data', 'w')

for (dirpath, dirnames, filenames) in os.walk(path):
    total_files = len(filenames)

    for filename in filenames:
        if filename == '.DS_Store':
            continue

        filedir = os.path.join(dirpath, filename)
        category = filedir.split('/')[-2]

        with open(filedir, 'r') as content_file:
            content = content_file.read()
            file_output.write(category + ', ' + repr(content))
            file_output.write('\n')

print 'Converted ' + str(total_files) + ' documents into single file \'20news.data\''
