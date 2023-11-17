import numpy as np
import math
from PIL import Image

n=0.3
# количество нейронов в скрытом слое
num_=3

w=0.3*np.random.rand(num_,65)
w_in=0.3*np.random.rand(2,num_)

#загружаем картинки
num_of_img=40
X=np.zeros((num_of_img,65)).astype(int)
Y=np.zeros(num_of_img).astype(int)
L=np.array([[0,0],[0,1],[1,0],[1,1]])
let=["A","B","C","D"]
for i in range(1,num_of_img+1):
    mas=np.zeros((8,8)).astype(int)
        
    folder="let/"+ str(i) + ".png"
    img = Image.open(folder)
    obj=img.load()
    for row in range(img.size[0]):
        for col in range(img.size[1]):
            p = obj[row,col]
            #определяем цвет
            if p[0]==255:
                mas[col][row]=0
            else:
                mas[col][row]=1
    # массив + фиктивный признак
    X[i-1]=np.append(mas.reshape(64),1)
    Y[i-1]=int((i-1)/10)

def net(x,_w):      # сеть
    return np.dot(x,_w)

def o(_net):        # cигмоида
    return 1/(1+math.exp(-_net))

def o_dif(_o):      # производная от сигмоиды
    return _o*(1-_o)

num_epoches=10000

for epoche in range(1,num_epoches):
    max_error=0
    for j in range(num_of_img):        
        x=X[j]
        letter=Y[j]

        error1=np.zeros(num_)
        error2=np.zeros(2)
        out1=np.zeros(num_)
        out2=np.zeros(2)
        
        # по скрытым сетям
        for i in range(num_):
            #совокупный вход и выход
            net_=net(x,w[i])
            
            out1[i]=o(net_)        
        
        # получаем выходы
        for i in range(2):
            #совокупный вход и выход
            net_=net(out1,w_in[i])
            out2[i]=o(net_)            
                
        # ошибка для каждого выходного элемента
        for i in range(2):
            
            error2[i]=(L[letter][i]-out2[i])*o_dif(out2[i])
            # находим максимальную ошибку
            if max_error<error2[i]:
                max_error=error2[i]  
                
        er=error2[0]+error2[1]
        if max_error<er:
                max_error=er  

                
        # ошибка для скрытого слоя
        for i in range(num_):            
            error1[i]=o_dif(out1[i])*(error2[0]*w_in[0][i]+error2[1]*w_in[1][i])
        
        # обновляем веса
        for i in range(65):
                for l in range(num_):
                    w[l][i]=w[l][i]+n*error1[l]*x[i]
                    
        
        for i in range(num_):
            for k in range(2):
                w_in[k][i]=w_in[k][i]+n*error2[k]*out1[i]
            
       

    # если ошибка меньше, то выходим из цикла
    #print(max_error)
    if(max_error<0.001):  
        print(epoche)
        print(max_error)
        break
        
        
# считываем букву


    mas=np.zeros((8,8)).astype(int)
        
    folder="let/"+ "41" + ".png"
    img = Image.open(folder)
    obj=img.load()
    for row in range(img.size[0]):
        for col in range(img.size[1]):
            p = obj[row,col]
            #определяем цвет
            if p[0]==255:
                mas[col][row]=0
            else:
                mas[col][row]=1
    # массив + фиктивный признак
    X_=np.append(mas.reshape(64),1)


letter_cur=0
x=X_
error1=np.zeros(num_)
error2=np.zeros(2)
out1=np.zeros(num_)
out2=np.zeros(2)
    
# по скрытым сетям
for i in range(num_):
    #совокупный вход и выход
    net_=net(x,w[i])
    out1[i]=o(net_)
            
# получаем выходы
for i in range(2):
    #совокупный вход и выход
    net_=net(out1,w_in[i])
    out2[i]=o(net_)
print(out2)
for i in range(2):
    if out2[i]<0.5:
        out2[i]=0
    else:
        out2[i]=1
for i in range(4):
    if out2[0]==L[i][0] and out2[1]==L[i][1]:
        print(let[i],let[Y[letter_cur]])



    

    

        
        
        
    
