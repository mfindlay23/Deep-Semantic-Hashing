import os
import sys

path = sys.argv[1]
output = sys.argv[2]

total_files = 0
file_output = open('20news.data', 'w')

i = 0
for (dirpath, dirnames, filenames) in os.walk(path):
    total_files = len(filenames)

    for filename in filenames:
        i = i + 1
        if filename == '.DS_Store':
            continue

        filedir = os.path.join(dirpath, filename)
        category = filedir.split('/')[-2]
        with open(filedir, 'r') as content_file:
            content = repr(content_file.read())[500:]
            file_output.write(category + ', ' + content)
            file_output.write('\n')

print i
print 'Converted ' + str(total_files) + ' documents into single file \'20news.data\''
