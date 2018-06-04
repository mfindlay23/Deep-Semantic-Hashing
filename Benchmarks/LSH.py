import binascii
import random

num_hashes = 10
num_docs = 21
shingle_length = 8
data_file = 'documents.txt'
doc_shingle_sets = {};
next_prime = 23 # http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php

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
  docs.append(doc_count)

  shingles_in_doc = set()
  shingles = [doc[i: i + shingle_length] for i in range(len(doc))][: -shingle_length]

  for shingle in shingles:
      crc = binascii.crc32(shingle) & 0xffffffff
      shingles_in_doc.add(crc)
      all_shingles.add(crc)

  doc_shingle_sets[doc_count] = shingles_in_doc
  doc_count += 1
  
data_file.close()  


a = generate_hash(num_hashes)
b = generate_hash(num_hashes)

sigs_list = []
for doc_id in docs:
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

estJSim = [0 for x in range(int(num_docs * (num_docs - 1) / 2))]

for row in range(num_docs):
  sig1 = sigs_list[row]

  for col in range(row + 1, num_docs):
    sig2 = sigs_list[col]

    sim = 0
    for z in range(num_hashes):
      sim = sim + (sig1[z] == sig2[z])
       
    estJSim[get_index(row, col)] = (sim / num_hashes)


# threshold = 0.8 
# print "\nList of Document Pairs with J(d1,d2) more than", threshold
# print "Values shown are the estimated Jaccard similarity and the actual"
# print "Jaccard similarity.\n"
# print "                   Est. J   Act. J"

# # For each of the document pairs...
# for i in range(0, num_docs):  
#   for j in range(i + 1, num_docs):
#     # Retrieve the estimated similarity value for this pair.
#     estJ = estJSim[get_index(i, j)]
    
#     # If the similarity is above the threshold...
#     if estJ > threshold:
    
#       # Calculate the actual Jaccard similarity for validation.
#       s1 = docs_as_shingle_sets[doc_names[i]]
#       s2 = docs_as_shingle_sets[doc_names[j]]
#       J = (len(s1.intersection(s2)) / len(s1.union(s2)))
      
#       # Print out the match and similarity values with pretty spacing.
#       print "  %5s --> %5s   %.2f     %.2f" % (doc_names[i], doc_names[j], estJ, J)
      