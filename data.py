import numpy as np
import re
import csv
import numpy as np
from sklearn.decomposition import TruncatedSVD


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

    # label = []
    # sentences = []
    # users = []
    #
    # # csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    # with open('train.tsv', 'r', encoding='utf-8') as file:
    #     file_list = tsv.TsvReader(file)
    #     m = 0
    #     for line in file_list:
    #         m = m + 1
    #         line = list(line)
    #         if line[1] == 'pants-fire' or line[1] == 'false' or line[1] == 'barely-true':
    #             label = -1
    #         else:
    #             label = 1
    #         # sentences.append(clean_str(line[2]))
    #真实的新闻和虚假的新闻分类
    #         users.append(line[4])
    # users_list = set(users)创建一个空的集合
    # users_counts = []
    # for user in users_list:
    #     count = 0
    #     for news in users:
    #         if news == user:
    #             count = count + 1
    #     users_counts.append([user, count])
    #用户发的新闻的数量
    # with open('user_list.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     for line in users_counts:
    #         writer.writerow(line)
    # with open('w2v.txt', 'a', encoding='utf-8') as file:
    #     for line in sentences:
    #        file.write(line+'\n')
    # with open('train.csv', 'r') as file:
    # sentences = word2vec.LineSentence("w2v.txt")
    # model = Word2Vec(sentences, window=3, size=32, sg=1, iter=8, negative=5, alpha=0.01)
    # model.wv.save_word2vec_format("word2vec.txt")

    #user_list_1 = []
    #with open('user_list_10.csv', 'r', newline='',encoding='utf-8') as file:
    #     reader = csv.reader(file)
    #    count = 0
    #    print(reader)
    #     #for i in len(reader):


         #for i in set(column1):
         #    user_list_1[i]=column1.count(i)
         #    num = eval(line[0])#if num >= 2:
         #    user_list_1.append([line[0], num])
    '''print(column1)
    with open('user_list_10.txt', 'a',newline='',encoding='utf-8') as f:
        for response in column1:
            f.write(response + '\n')'''

    #np.savetxt('user_list_1.txt',user_list_1)
    #with open('user_list_1.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     for line in user_list_1:
    #         writer.writerow(line)
    #user_list = []
    #with open('user_list.csv', 'r') as file:
    #     reader = csv.reader(file)
    #     for line in reader:
    #         user_list.append(line[0])
    #user_num = len(user_list)
    #news_num = 10269
    # news_user_matrix = np.zeros((news_num, user_num))
    # label = np.zeros((news_num,))
    # with open('news_user.csv', 'r') as file:
    #     reader = csv.reader(file)
    #     row = 0
    #     for line in reader:
    #         column = user_list.index(line[0])
    #         # label[row] = eval(line[1])
    #         news_user_matrix[row, column] = 1
    #         row = row + 1
    # np.savetxt("news_user_matrix.txt", news_user_matrix, fmt='%d')
    #user_list_1 = []
    #with open('user_list_1.csv', 'r') as file:
    #   reader = csv.reader(file)
    #    for line in reader:
    #        user_list_1.append(line[0])
    # news1_index = []
    # label_list = []
    # news_user_1_list = []
    # with open('news_user.csv', 'r') as file:
    #     reader = csv.reader(file)
    #     index = 0
    #     for line in reader:
    #         user_name = line[0]
    #         label = eval(line[1])
    #         if user_name in user_list_1:
    #             news1_index.append(index)
    #             news_user_1_list.append([user_name, label])
    #             label_list.append(label)
    #         index = index + 1
    # with open('news_user_1.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     for line in news_user_1_list:
    #         writer.writerow(line)
    with open('first.txt', 'r') as f:
        reader_sum = csv.reader(f)
        reader_sum = list(reader_sum)
    news_user_matrix_1 = np.zeros((5498, 111))
    print(news_user_matrix_1.shape)
    with open('user_list_10.txt', 'r') as file:
        reader = csv.reader(file)
        row = 0
        reader =list(reader)
        print(type(reader))
        print(len(reader))
        for i in range(5498):
            for j in range(111):
                if reader[i] == reader_sum[j]:
                    news_user_matrix_1[i][j]=1
        print(news_user_matrix_1.shape)
        np.savetxt("news_user_matrix_2.txt", news_user_matrix_1, fmt='%d')
        a = np.loadtxt('news_user_matrix_1.txt')
        print(a.shape)
        b=np.loadtxt("label_4.txt")
        print(b)
        b[np.where(b==0)]=-1
        np.savetxt("label_5.txt",b, fmt='%d')
        '''for line in  reader:
            column = user_list_1.index(line[0])
            # label[row] = eval(line[1])
            news_user_matrix_1[row, column] = 1
            row = row + 1
    np.savetxt("news_user_matrix_1.txt", news_user_matrix_1, fmt='%d')'''



