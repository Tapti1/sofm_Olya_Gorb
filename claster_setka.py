import numpy as np
import matplotlib.pyplot as plt
import random
import math

# датасет
X= np.zeros([66,2])
r=8
for i in range(32):
    X[i*2][0]=i/4
    X[i*2][1]=math.sqrt(r**2-(X[i*2][0])**2)+r
    
    X[i*2+1][0]=i/4
    X[i*2+1][1]=-math.sqrt(r**2-(X[i*2+1][0])**2)+r
    
# задаём рандомно веса
w=np.random.rand(8,16,2)
a=4
for i in range(w.shape[0]):
    for j in range(w.shape[1]):
        w[i][j]=np.array([i/2,j/2])   
   
def sosed(epoche,i,j,i_win,j_win):     # функция соседства
    dist=math.sqrt((i_win-i)**2+(j_win-j)**2);
    h=np.exp(-(dist**2)/(2*(u(epoche)**2))) # функция растояния
    sigma=epoche**(-0.2)    # функция скорости обучения
    return h*sigma
    
def u(epoche):      # радиус обучения
    return a*epoche**(-0.2)
    

for i in range(1,100):  
    #берём рандомную точку    
    r_point=X[random.randint(0, X.shape[0]-1)]

    
    # находим ближайшую к ней точку
    dist=1000000000
    p=np.array([0,0])
    p_=np.array([0,0])
    for ii in range(w.shape[0]):
        for jj in range(w.shape[1]): 
            if np.linalg.norm(r_point-w[ii][jj])<dist:
                dist=np.linalg.norm(r_point-w[ii][jj])
                p=w[ii][jj]  
                p_=[ii,jj]
   
    # меняем веса     
    for ii in range(w.shape[0]):        
        for jj in range(w.shape[1]): 
            #print(sosed(i,ii,jj,p_[0],p_[1]))
            w[ii][jj]+=sosed(i,ii,jj,p_[0],p_[1])*(r_point-w[ii][jj])   

    plt.plot(w[:,:,0], w[:,:,1])
    for i in range(w.shape[0]):
        plt.plot(w[i,:,0], w[i,:,1])
    plt.scatter(w[:,:,0], w[:,:,1])
    plt.scatter(X[:, 0], X[:, 1],s=10,alpha=0.5)
    plt.show()

#рисуем график
    plt.plot(w[:,:,0], w[:,:,1])
    for i in range(w.shape[0]):
        plt.plot(w[i,:,0], w[i,:,1])
plt.scatter(w[:,:,0], w[:,:,1])
plt.scatter(X[:, 0], X[:, 1], s=10,alpha=0.5)
plt.show()



