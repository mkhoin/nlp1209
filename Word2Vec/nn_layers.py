#!/usr/bin/env python
# coding: utf-8

# ## Repeat 노드

# In[1]:


import numpy as np

# D, N = 8, 7
# x = np.random.rand(1, D)
# print(x,x.shape)   # (1,8)
# print('-'*70)
# y = np.repeat(x, N, axis=0)  # 수직(행) 방향으로 N(7)번 반복 생성
# print(y,y.shape)   # (7,8)


# In[2]:


# dy = np.random.rand(N, D)
# print(dy,dy.shape)  # (7,8)
# dx = np.sum(dy, axis=0, keepdims=True)  # 수직방향 합, keepdims=True이면 2차원, False면 1차원
# print('-'*70,'\n',dx,dx.shape)


# ## Sum 노드

# In[3]:


# import numpy as np

# D, N = 8, 7
# x = np.random.rand(N, D)
# print(x)
# print('\n')
# y = np.sum(x, axis=0, keepdims=True)
# print(y)
# # dy = np.random.rand(1, D)


# In[4]:


# dx = np.repeat(y, N, axis=0)
# dx


# ## MatMul 노드

# In[5]:


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW  # 깊은 복사
        return dx


# In[6]:


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a = b
# print(a)
id(a) == id(b)


# In[7]:


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a[...] = b
# print(a)
id(a) == id(b)


# ## 시그모이드 계층

# In[8]:


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


# ### ReLU 계층

# In[9]:


class ReLU:
    def __init__(self):
        self.params, self.grads = [], []
        self.mask = None
        self.out = None

    def forward(self, x):
        self.mask = (x <= 0)  # x가 0 이하일 경우 0으로 변경
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0  #  x가 0 이하일 경우 0으로 변경
        dx = dout 
        return dx


# ## Affine 계층 : MatMul 노드 에 bias를  더한 계층 

# In[10]:


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


# In[11]:


a = np.zeros_like([[1,2,3]]) # 인자와 shape이 같은 배열을 모든 요소를 0으로 생성
# print(a,type(a))


# ## Softmax with Loss 계층

# In[12]:


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


# def relu(x):
#     return np.maximum(0, x)


# def softmax(x):
#     if x.ndim == 2:
#         x = x - x.max(axis=1, keepdims=True)
#         x = np.exp(x)
#         x /= x.sum(axis=1, keepdims=True)
#     elif x.ndim == 1:
#         x = x - np.max(x)
#         x = np.exp(x) / np.sum(np.exp(x))

#     return x


# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
        
#     # 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
#     if t.size == y.size:
#         t = t.argmax(axis=1)
             
#     batch_size = y.shape[0]

#     return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블
        
    def softmax(self,x):
        if x.ndim == 2:
            x = x - x.max(axis=1, keepdims=True)
            x = np.exp(x)
            x /= x.sum(axis=1, keepdims=True)
        elif x.ndim == 1:
            x = x - np.max(x)
            x = np.exp(x) / np.sum(np.exp(x))
        return x 
    
    def cross_entropy_error(self,y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]

        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
      

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)

        # 정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = self.cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


# In[13]:


# a = np.exp(1)  # e^1, e = 2.718281828459045
# print(a)
# print(2.718281828459045**1)


# ## 가중치 갱신

# In[14]:


class SGD:
    '''
    확률적 경사하강법(Stochastic Gradient Descent)
    '''
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


# In[15]:


class Adam:
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    
    '''
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
            
# https://dalpo0814.tistory.com/29

import time

def remove_duplicate(params, grads):
    '''
    매개변수의 중복 제거 함수
    매개변수 배열 중 중복되는 가중치를 하나로 모아
    그 가중치에 대응하는 기울기를 더한다.
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유 시
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 경사를 더함
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # 뒤섞기
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # 기울기 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                
                params, grads = remove_duplicate(model.params, model.grads)  # 공유된 가중치를 하나로 모음
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 손실 %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('반복 (x' + str(self.eval_interval) + ')')
        plt.ylabel('손실')
        plt.show()
       



    
# Embedding 계층
class Embedding :
    def __init__(self,W):
        self.params =[W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
    
    # 순전파
    def forward(self,idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    # 역전파 
    def backward(self,dout):   
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None
    

 # EembeddingDot 계층
class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):  
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1) # 

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh    

import collections
class UnigramSampler:
    
    # 생성자 : corpus를 사용하여 단어의 0.75제곱 처리한 확률 분포를 구함
    def __init__(self, corpus, power, sample_size): # power= 0.75, sample_size = 2
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        # corpus 내의 단어별 발생횟수를 구함    
        counts = collections.Counter()  
        for word_id in corpus:   # corpus: [0 1 2 3 4 1 5 6], 
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size  # 7

        self.word_p = np.zeros(vocab_size)  # (7,)
        for i in range(vocab_size):  # 7 
            self.word_p[i] = counts[i]  # [1, 2, 1, 1, 1, 1, 1] ,단어 발생 횟수

        self.word_p = np.power(self.word_p, power) # 0.75제곱
        self.word_p /= np.sum(self.word_p)  # 전체의 합으로 나누어 확률을 구함

    def get_negative_sample(self, target):   # target = np.array([1, 3, 0]), (3,)
        batch_size = target.shape[0]  # 3
        
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)  # (3,2)

        for i in range(batch_size):  # 2회
            p = self.word_p.copy()
            target_idx = target[i]  # 1,3,0
            p[target_idx] = 0  # p[1]=p[3]=p[0]=0
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
            
        return negative_sample
    

# SigmoidWithLoss 클래스 사용 
class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid의 출력
        self.t = None  # 정답 데이터

    def cross_entropy_error(self,y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]

        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
        
    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))   # sigmoid , 예측값

        self.loss = self.cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx

    
# NegativeSamplingLoss 클래스

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):  #  sample_size : 부정적 예 샘플링 수 (2개)
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]     # 3개층, 긍정 1층 + 부정 2층
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]  # 3개층, 긍정 1층 + 부정 2층

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target): # h: 은닉층 뉴런수, target은 긍정적 예의 target
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target) # 부정적 예

        # 긍정적 예 순전파
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)  # 값이 모두 1 : 긍정
        loss = self.loss_layers[0].forward(score, correct_label)

        # 부정적 예 순전파
        negative_label = np.zeros(batch_size, dtype=np.int32) # 값이 모두 0 : 부정
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1): # 입력값을 각 계층의 backward만 호출하여 전달
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers): # 역전파이므로  los_layer가 먼저 호출된다
            dscore = l0.backward(dout) # SigmoidWithLoss 계층
            dh += l1.backward(dscore)  # EmbeddingDot 계층

        return dh