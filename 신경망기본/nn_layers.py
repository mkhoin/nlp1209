#!/usr/bin/env python
# coding: utf-8

# ## Repeat 노드

# In[1]:


import numpy as np

# 순전파

D = 8
N = 7
x = np.random.rand(1,D)  # (1,8)
print(x,x.shape)
print('-'*70)
y = np.repeat(x,N,axis=0) # 수직(행) 방향, axis=0
print(y,y.shape)  # (7, 8)


# In[2]:


# 역전파
dy = np.random.rand(N,D)
print(dy,dy.shape) # (7,8)
print('-'*70)
dx = np.sum(dy,axis=0,keepdims=True) # 수직방향 합, keepdims=True이면 2차원, False면 1차원
print(dx,dx.shape)  # (1, 8)


# ## Sum 노드

# In[3]:


# 순전파

D,N = 8,7
x = np.random.rand(N,D)  # (7,8)
print(x,x.shape)
print('-'*70)
y = np.sum(x,axis=0,keepdims=True) # 수직방향 합, keepdims=True이면 2차원, False면 1차원
print(y,y.shape)  # (1, 8)


# In[4]:


#역전파

D = 8
N = 7
x = np.random.rand(1,D)  # (1,8)
print(x,x.shape)
print('-'*70)
y = np.repeat(x,N,axis=0) # 수직(행) 방향, axis=0
print(y,y.shape)  # (7, 8)


# ## MatMul 노드

# In[5]:


class MatMul:
    def __init__(self,W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
        
    def forward(self,x):
        W = self.params
        out = np.dot(x,W)
        self.x = x
        return out
    
    def backward(self,dout):
        W = self.params
        dx = np.dot(dout,W.T)
        dW = np.dot(self.x.T,dout)
        self.grads[0][...] = dw  # 깊은 복사
        return dx        


# In[6]:


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a = b
print(a)
id(a) == id(b)


# In[7]:


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a[...] = b
print(a)
id(a) == id(b)


# In[8]:


# np.zeros_like
a = np.arange(12).reshape(3,4)
b = np.zeros_like(a)
b


# ## 시그모이드 계층

# In[9]:


class Sigmoid:
    def __init__(self):
        self.params,self.grads = [],[]
        self.out = None
        
    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self,dout):
        dx = dout * self.out * (1 - self.out)  # 공식 도출은 참고서적 참조
        return dx


# ## Affine 계층 : MatMul 노드 에 bias를  더한 계층 

# In[10]:


class Affine:
    def __init__(self,W,b):
        self.params = [W,b]
        self.grads = [np.zeros_like(W),np.zeros_like(b)]
        self.x = None
        
    def forward(self,x):
        W, b = self.params
        out = np.dot(x,W) + b
        self.x = x
        return out 
    
    def backward(self,dout):
        W,b = self.params
        dx = np.dot(dout,W.T)
        dW = np.dot(self.x.T,dout)
        db = np.sum(dout,axis=0)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


# In[11]:


a = np.zeros_like([[1,2,3]]) # 인자와 shape이 같은 배열을 모든 요소를 0으로 생성
print(a,type(a))


# ## Softmax with Loss 계층

# In[12]:


class SoftmaxWithLoss:
    def __init__(self):
        self.params,self.grads = [], []
        self.y = None    # softmax의 출력 값
        self.t = None    # 정답 레이블
        
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
     
    def backward(self,dout=1):
        batch_size = self.t.shape[0]

        # dx = (self.y - self.t)/batch_size # 순수 Softmax계층 일경우
        
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size
        return dx


# ## 가중치 갱신

# In[13]:


# 확률적 경사하강법(Stochastic Gradient Descent)
class SGD :
    def __init__(self,lr=0.01):
        self.lr = lr
        
    def update(self,params,grads):
        for i in range(len(params)):
            params[i] -= self.lr*grads[i]


# In[14]:


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

