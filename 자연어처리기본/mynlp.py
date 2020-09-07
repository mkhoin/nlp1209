#!/usr/bin/env python
# coding: utf-8

# ### 파이썬으로 말뭉치 전처리 하기 : 통계 기반 기법
# 
# #### 말뭉치 또는 코퍼스(corpus)는 자연언어 연구를 위해 특정한 목적을 가지고 언어의 표본을 추출한 집합이다.
# #### 대량의 텍스트 데이터
# 컴퓨터의 발달로 말뭉치 분석이 용이해졌으며 분석의 정확성을 위해 해당 자연언어를 형태소 분석하는 경우가 많다. 확률/통계적 기법과 시계열적인 접근으로 전체를 파악한다. 언어의 빈도와 분포를 확인할 수 있는 자료이며, 현대 언어학 연구에 필수적인 자료이다. 

# In[1]:


text = 'You say goodbye and I say hello.'


# In[2]:


text = text.lower()


# In[3]:


text = text.replace('.', ' .')


# In[4]:


text


# In[5]:


words = text.split(' ')


# In[6]:


words


# In[7]:


list(set(words))  # 중복된 단어 제거


# ###  딕셔너리를 이용하여 단어 ID와 단어를 짝지어 주는 대응표 작성

# In[8]:


word_to_id = {}
id_to_word = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word


# In[9]:


id_to_word


# In[10]:


word_to_id


# In[11]:


id_to_word[1]


# In[12]:


word_to_id['hello']


# In[13]:


import numpy as np
corpus = [word_to_id[w] for w in words]
corpus = np.array(corpus)
corpus


# ###  말뭉치를 이용하기 위한 전처리 함수 구현

# In[14]:


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


# In[15]:


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)


# In[16]:


corpus


# In[17]:


word_to_id


# In[18]:


id_to_word


# ### 동시발생 행렬 (Co-occurence Matrix)

# In[19]:


# import sys
# sys.path.append('..')
# import numpy as np
# from common.util import preprocess
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)


# In[20]:


print(corpus)


# In[21]:


print(id_to_word)


# In[22]:


C = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
], dtype=np.int32)


# In[23]:


print(C[0])  # id 가 0 인 단어의 벡터 표현


# In[24]:


print(C[4])  # id 가 4 인 단어의 벡터 표현


# In[25]:


word_to_id['goodbye'] # 2


# In[26]:


print(C[word_to_id['goodbye']])  # 'goodbye'의 벡터 표현


# In[27]:


# 동시발생 행렬을 생성하는 함수
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


# In[28]:


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
C


# In[29]:


text = 'I like apple and you like banana.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
C


# ### 벡터 간 유사도 : 코사인 유사도(Cosine Similarity)

# In[30]:


def cos_similarity(x, y):
    nx = x / np.sqrt(np.sum(x ** 2))
    ny = y / np.sqrt(np.sum(y ** 2))
    return np.dot(nx, ny)
# 입력 인수로 제로 벡터(원소가 모두 0인 벡터)가 들어오면 'divide by zero' 오류 발생


# In[31]:


# 개선된 코싸인 유사도 : 작은 값 eps(엡실론)을 분모에 더해준다, 부동소수점 게산시 반올림되어 다른 값에 흡수된다
def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


# In[32]:


# import sys
# sys.path.append('..')
# from common.util import preprocess, create_co_matrix, cos_similarity


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']]  # "you"의 단어 벡터
c1 = C[word_to_id['i']]    # "i"의 단어 벡터
print(c0)
print(c1)
print(cos_similarity(c0, c1))


# In[33]:


text = 'I like apple and you like banana.'
# text = 'I hate apple and you dislike banana.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['apple']]  
c1 = C[word_to_id['banana']]
# c0 = C[word_to_id['hate']]  
# c1 = C[word_to_id['dislike']]
print(c0)
print(c1)
print(cos_similarity(c0, c1))


# ### 유사 단어의 랭킹 표시

# In[34]:


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print('%s(을)를 찾을 수 없습니다.' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 코사인 유사도 계산
    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


# In[35]:


x = np.array([100, -20, 2])


# In[36]:


x.argsort()


# In[37]:


(-x).argsort()


# In[38]:


# import sys
# sys.path.append('..')
# from common.util import preprocess, create_co_matrix, most_similar


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C, top=5)


# In[ ]:




