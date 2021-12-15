import numpy as np
import re
import csv
import numpy as np
from sklearn.decomposition import TruncatedSVD
import tsv


def clean_str(str):
    str = re.sub("[^A-Za-z0-9(),!?'`]", " ", str)

    str = re.sub(" can't ", " can n't ", str)
    str = re.sub("'", " ' ", str)
    str = re.sub("\s{2,}", " ", str)
    str = re.sub("' s ", " 's ", str)
    str = re.sub("' ve ", " 've ", str)
    str = re.sub("' re ", " 're ", str)
    str = re.sub("' d ", " 'd ", str)
    str = re.sub("' m ", " 'm ", str)
    str = re.sub("n ' t ", " n't ", str)
    str = re.sub("' ll ", " 'll ", str)
    str = re.sub("\(", " ( ", str)
    str = re.sub("\)", " ) ", str)
    str = re.sub("\?", " ? ", str)
    str = re.sub("!", " ! ", str)
    str = re.sub(", ", " , ", str)

    str = re.sub("\s{2,}", " ", str)


    return str.strip().lower()


if __name__ == "__main__":
    doc=[]
    with open('w2v.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line=clean_str(line)
            doc.append(line)
    with open('www2vvv.txt', 'w', encoding='utf-8') as file1:
        for i in doc:
            file1.write(str(i))
            file1.write("\n")
        file1.close()
