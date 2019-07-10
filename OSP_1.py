import scipy.io
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import cv2

def Read_Dara(DATA):
    mat = scipy.io.loadmat(DATA)
    a=mat['cube']
    return a

def Make_D_U(d_number,d_posit,u_number,u_posit):


    u_posit_matrix=np.zeros([u_number,2])
    d_posit_matrix=np.zeros([d_number,2])

    u=np.zeros([a.shape[2],u_number])
    d=np.zeros([a.shape[2],d_number])

    for k in range(u_number):
        for i in range(len(a[1,1,:])):
            u[i][k]=a[u_posit[k*2],u_posit[k*2+1],i]

    for k in range(d_number):
        for i in range(len(a[1,1,:])):
            d[i][k]=a[d_posit[k*2],d_posit[k*2+1],i]

    print('U_shape',u.shape)
    print('d_shape',d.shape)

    return u,d

def Make_PuT(u):

    uT=np.transpose(u)
    uTu=uT.dot(u)
    uTu_inverce=inv(uTu)
    # print(uTu_inverce)
    uTu_inverce_uT=uTu_inverce.dot(uT)
    id_matrix=np.identity(256)
    u_uTu_invers_uT=u.dot(uTu_inverce_uT)
    PuT=id_matrix-u_uTu_invers_uT    #identity matrix - pseudo_inverse
    return PuT

def OSP(PuT,x,y,INPUT):

    inputdata=INPUT[x][y]
    # print('input',inputdata.shape)
    PuTR=np.transpose(inputdata.dot(PuT)) # R the spectral (PuT*R)
    # print('PuTR', PuTR.shape)
    PuTR_TR=np.transpose(d) # transpose of D
    OSP_result=PuTR_TR.dot(PuTR) #the result of mix3
    # print('OSP_result',OSP_result.shape)
    return OSP_result

def INPUT_Matrix(a):
    x_pixel=len(a[:,1,1])
    y_pixel=len(a[1,:,1])
    z_pixel=len(a[1,1,:])
    INPUT=np.zeros([x_pixel,y_pixel,z_pixel])
    print('INPUT=',INPUT.shape)
    for i in range(len(a[:,1,1])):
        for j in range(len(a[1,:,1])):
            for k in range(len(a[1,1,:])):
                INPUT[i][j][k]=a[i,j,k]
    # print(INPUT)
    return INPUT

def Make_histogram(ans):
    # histo = np.histogram(ans, bins=np.arange(0, 1, 0.001))
    # print(histo)
    plt.hist(ans, bins=np.arange(-1, 1, 0.01))
    plt.show()

def SHOW_result(ans):
    plt.figure()
    plt.imshow(a[:, :, 50])

    cv2.imshow('thermal figure ', ans)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    DATA="2-ck-1_20190417115056_wb.mat"
    a=Read_Dara(DATA)
    u_number=6
    u_posit=[174,75,162,42,145,234,92,250,195,181,80,133]
    d_number = 1
    d_posit = [171, 163]
    u,d=Make_D_U(d_number,d_posit,u_number,u_posit)
    PuT = Make_PuT(u)
    INPUT=INPUT_Matrix(a)
    print(INPUT.shape)
    ans=np.zeros([a.shape[0],a.shape[1]])
    print('ans=',ans.shape)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ans[i][j]=OSP(PuT,i,j,INPUT)

    # Make_histogram(ans)

    for x in range(ans.shape[0]):
        for y in range(ans.shape[1]):
            if ans[x][y]>=0.32:
                ans[x][y]=0
            else:
                ans[x][y]=255


    SHOW_result(ans)