import os

path = "c:\\python25"

i = 0
for (path, dirs, files) in os.walk(pp, followlinks=True):
    print(path)
    print(dirs)
    print(files)
    print("----")
    i += 1
    if i >= 4:
        break