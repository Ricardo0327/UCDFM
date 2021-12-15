import wordninja
from nltk.tokenize import RegexpTokenizer
doc_set=[]
with open('w2v.txt','r',encoding='utf8' ) as f:
    for line in f:
        tokens = wordninja.split(line)
        doc_set.append(tokens)

with open('ww2vv.txt', 'w', encoding='utf-8') as file1:
    for i in doc_set:
        file1.write(str(i))
        file1.write("\n")
    file1.close()