import binascii
import random
import sys

num_hashes = 128
num_docs = 11314
shingle_length = 8
data_file = sys.argv[1]
doc_shingle_sets = {}
doc_id_to_category = {}
next_prime = 11317 # http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php

def get_index(row, col):
  if row == col:
    sys.stderr.write("Incorrect Access")
    sys.exit(1)

  if col < row:
    temp =row
    row = col
    col = temp

  return int(row * (num_docs - (row + 1) / 2.0) + col - row) - 1

def generate_hash(k):
  # Create a list of 'k' random values.
  rand_list = []

  while k > 0:
    # Get a random shingle ID.
    rand_idx = random.randint(0, doc_count)
    # Ensure that each random number is unique.
    while rand_idx in rand_list:
      rand_idx = random.randint(0, doc_count)

    # Add the random number to the list.
    rand_list.append(rand_idx)
    k = k - 1
  return rand_list

data_file = open(data_file, "rU")

docs = []
doc_count = 0
all_shingles = set()

for i in range(0, num_docs):
  doc = data_file.readline()
  category = doc.split(',')[0]
  doc = doc.split(',')[1:]
  doc = ','.join(doc)
  doc = doc.split('\'')[1:-1]
  doc = '\''.join(doc)
  doc = str(doc)
  docs.append(doc_count)

  shingles_in_doc = set()
  shingles = [doc[i: i + shingle_length] for i in range(len(doc))][: -shingle_length]

  for shingle in shingles:
      crc = binascii.crc32(shingle) & 0xffffffff
      shingles_in_doc.add(crc)
      all_shingles.add(crc)

  doc_shingle_sets[doc_count] = shingles_in_doc
  doc_id_to_category[doc_count] = category
  doc_count += 1

data_file.close()

a = generate_hash(num_hashes)
b = generate_hash(num_hashes)

sigs_list = []
for i, doc_id in enumerate(docs):
  shingle_id_set = doc_shingle_sets[doc_id]

  doc_sig = []
  for i in range(0, num_hashes):
    min_hash_code = next_prime + 1

    for shingle_id in shingle_id_set:
      hash_code = (a[i] * shingle_id + b[i]) % next_prime

      if hash_code < min_hash_code:
        min_hash_code = hash_code

    doc_sig.append(min_hash_code)
  sigs_list.append(doc_sig)

file_output = open('hash_codes.out', 'w')
for i, hash_code in enumerate(sigs_list):
  if len(hash_code) != num_hashes:
    print 'Bad hash code'
  file_output.write(str(hash_code) + ', ' + str(doc_id_to_category[docs[i]]) + '\n')
