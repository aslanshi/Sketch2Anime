import os

img_path = 'E:/Code/Anime Faces/out/'
f = open("name_file.txt", "w")

names = os.listdir(img_path)

print(len(names))

for idx in range(len(names)):
    if idx == 0:
        f.write(names[idx])
    else:
        f.write(",{}".format(names[idx]))