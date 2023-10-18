import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import random

# датасет
X, y = datasets.make_blobs(n_samples=50, random_state=3)
# задаём рандомно веса
w=np.random.rand(2,4)*5
a=2000

def sosed(epoche,dist):     # функция соседства
    h=np.exp(-dist*dist/2*u(epoche)) # функция растояния
    sigma=epoche**(-0.2)    # функция скорости обучения
    return h*sigma
    
def u(epoche):      # радиус обучения
    return a*epoche**(-0.2)
    

for i in range(1,100):
    
    # темп обучения
    n=i**(-0.2)
    
    #берём рандомную точку    
    r_point=X[random.randint(0, 49)]

    
    # находим ближайшую к ней точку
    dist=[np.linalg.norm(r_point-np.array([w[0][0],w[1][0]])),np.linalg.norm(r_point-np.array([w[0][1],w[1][1]])),np.linalg.norm(r_point-np.array([w[0][2],w[1][2]]))]
    min_cl=np.argmin(dist) 
    
    # меняем веса

    p=np.array([w[0][min_cl],w[1][min_cl]])
   
    for j in range(w.shape[1]):        
        distance = np.linalg.norm(p - np.array([w[0][j], w[1][j]])) 
        w[0][j]+=n*sosed(i,distance)*(r_point[0]-w[0][j])
        w[1][j]+=n*sosed(i,distance)*(r_point[1]-w[1][j])
        #print(w[1][j])
    
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.scatter(w[0], w[1])
    plt.show()

#рисуем график

plt.scatter(X[:, 0], X[:, 1], s=10)
plt.scatter(w[0], w[1])
plt.show()



