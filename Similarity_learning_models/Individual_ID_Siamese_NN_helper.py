# Importing necessary libraries
#  ===============================
from tensorflow.keras.layers import Activation,Add,BatchNormalization,Conv2D,Concatenate,Flatten,Dense
from tensorflow.keras.layers import Input,GlobalMaxPooling2D,MaxPooling2D,Lambda,Reshape,Dropout
from tensorflow.keras.preprocessing.image import img_to_array,array_to_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
import tensorflow_addons as tfa
from scipy.ndimage import affine_transform
from PIL import Image as pil_image
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import io
import os
import matplotlib.pyplot as plt

from Individual_ID_Triplet_NN_helper import*

# ======================================================


class ImsPairs(Sequence):

    
    def __init__(self, Ims, Labels, batch_size=128):
        super().__init__()
        self.Ims       = Ims
        self.Labels    = Labels
        self.batch_size = batch_size
          
    def __getitem__(self, index):
        start = self.batch_size*index
        size  = min(len(self.Ims) - start, self.batch_size)
        ImsA  = np.zeros((size,),dtype='<U24')
        ImsY  = np.zeros((size,))
        for i in range(size):
            ImsA[i] = self.Ims[start + i]
            ImsY[i]     = self.Labels[start+i]
            
        (ImL,ImR), Y = self.GenPairs(ImsA,ImsY)
      
        return ImL,ImR,Y
    
    def __len__(self):
        return (len(self.Ims) + self.batch_size - 1)//self.batch_size
    
    def GenPairs(self,Ims, IDs):
        
        x1,x2 = np.triu_indices(IDs.shape[0],1)
        m = np.where(IDs[x1]==IDs[x2])[0]
        u = np.where(IDs[x1]!=IDs[x2])[0]
        np.random.shuffle(u)
        uu = u[:m.shape[0]]
        y_p = np.ones((m.shape[0],))
        y_n = np.zeros(uu.shape[0],)
        idx = np.hstack([m,uu])
        y = np.hstack([y_p,y_n])
        idxL, idxR = x1[idx],x2[idx]
        Idx = np.arange(y.shape[0])
        np.random.shuffle(Idx)
        
        return (Ims[idxL[Idx]],Ims[idxR[Idx]]),y[Idx]
    
def Generator2Pairs(generator):
    x1,x2,y = zip(*generator)
    xx1 = np.concatenate(x1,axis=0)
    xx2 = np.concatenate(x2,axis=0)
    yy = np.concatenate(y,axis=0)
    return xx1,xx2,yy

# ==========================================================

class BatchGenerator(Sequence):

    
    def __init__(self, ImL,ImR,Y,pre,shp, batch_size=64):
        super(BatchGenerator).__init__()
        self.ImL = ImL
        self.ImR = ImR
        self.Y = Y
        self.pre = pre
        self.shp = shp
        self.batch_size = batch_size
          
    def __getitem__(self, index):
        start = self.batch_size*index
        size  = min(len(self.ImL) - start, self.batch_size)
        ImL  = np.zeros((size,) + self.shp, dtype=K.floatx())
        ImR  = np.zeros((size,) + self.shp, dtype=K.floatx())
        ImsY  = np.zeros((size,), dtype = K.floatx())
        for i in range(size):
            ImL[i,:,:,:] = self.pre.read_for_training(self.ImL[start + i])
            ImR[i,:,:,:] = self.pre.read_for_training(self.ImR[start + i])
            ImsY[i]     = self.Y[start+i]
    
        return (ImL,ImR),np.expand_dims(ImsY,axis=1)
    
    def __len__(self):
        return (len(self.ImL) + self.batch_size - 1)//self.batch_size
    
# ==================================================================

class ValGenerator(Sequence):

    
    def __init__(self, IsL,IsR,Labels,pre,shp,batch_size=64):
        super(BatchGenerator).__init__()
        self.IsL = IsL
        self.IsR = IsR
        self.Labels  = Labels
        self.pre = pre
        self.shp = shp
        self.batch_size = batch_size
        
          
    def __getitem__(self, index):
        start = self.batch_size*index
        size  = min(len(self.IsL) - start, self.batch_size)
        ImL  = np.zeros((size,) + self.shp, dtype=K.floatx())
        ImR  = np.zeros((size,) + self.shp, dtype=K.floatx())
        ImsY  = np.zeros((size,), dtype = K.floatx())
        for i in range(size):
            ImL[i,:,:,:] = self.pre.read_for_validation(self.IsL[start + i])
            ImR[i,:,:,:] = self.pre.read_for_validation(self.IsR[start + i])
            ImsY[i]     = self.Labels[start+i]
        
        
        return (ImL,ImR),ImsY
    
    def __len__(self):
        return (len(self.IsL) + self.batch_size - 1)//self.batch_size
    
# ====================================================================

def GenTestPairs(Ims, IDs,n):
    xl,xr = np.triu_indices(len(IDs),1)
    LIDs, RIDs = np.array(IDs)[xl], np.array(IDs)[xr]
    Id = np.where(LIDs==RIDs)[0]
    np.random.shuffle(Id)
    m = Id[:n]
    Ml, Mr, Lm = xl[m],xr[m], np.ones((len(m),))
    u = np.where(LIDs!=RIDs)[0]
    np.random.shuffle(u)
    new_u = u[:n]
    Ul, Ur, Lu = xl[new_u],xr[new_u], np.zeros(len(new_u),)
    XL, XR, L = np.hstack([Ml,Ul]),np.hstack([Mr,Ur]), np.hstack([Lm,Lu])
    Idx = np.arange(len(L))
    np.random.shuffle(Idx)
    return np.array(Ims)[XL[Idx]],np.array(Ims)[XR[Idx]],L[Idx]

# =============================================


class MYModel(object):
    
    def __init__(self,shp,lr,l2=0.0,k1=1,k2=2,k3=3,k9=9,mid =32):
        self.k1,self.k2,self.k3,self.k9,self.mid = k1,k2,k3,k9,mid
        self.regul = regularizers.l2(l2)
        self.shp = shp
        self.kwargs = {'padding':'same', 'kernel_regularizer':self.regul}
        self.optim, self.s2 = Adam(lr),self.k2
        self.InputIm = Input(shape = self.shp)
        self.BaseModel = self.BranchModel()
        self.HeadModel = self.TopModel()
        
        
    def BranchModel(self):
        x = self.FirstBlock(self.InputIm,64)
        x = self.ConvBlockSubblock(x,128,64)
        x = self.ConvBlockSubblock(x,256,64)
        x = self.ConvBlockSubblock(x,384,96)
        x = self.ConvBlockSubblock(x,512,128)
        x = GlobalMaxPooling2D()(x)
        return Model(self.InputIm, x, name = 'Base')
    
    def TopModel(self):
        Fl = Input(shape = self.BaseModel.output_shape[1:])
        Fr = Input(shape = self.BaseModel.output_shape[1:])
        x1 = self.SumF()([Fl,Fr])
        x2 = self.ProdF()([Fl,Fr])
        x3 = self.AbsDifF()([Fl,Fr])
        x4 = self.DifSqF()(x3)
        x  = Concatenate()([x1,x2,x3,x4])
        x  = Reshape((4, self.BaseModel.output_shape[1], 1), name='reshape1')(x)
        x  = self.PerFCNN(x)
        return Model([Fl,Fr], x, name = 'Head')
    

    def SiameseNet(self):
        ImL = Input(shape = self.shp)
        ImR = Input(shape = self.shp)
        FtL = self.BaseModel(ImL)
        FtR = self.BaseModel(ImR)
        Top   = self.HeadModel([FtL,FtR])
        model = Model([ImL,ImR],Top)
        model.compile(self.optim,loss = 'binary_crossentropy', metrics=['acc'])
        return model
    
    
    def Subblock(self,x, convF):
        x = BatchNormalization()(x)
        y = x
        y = Conv2D(convF, (self.k1, self.k1), activation='relu', **self.kwargs)(y) 
        y = BatchNormalization()(y)
        y = Conv2D(convF, (self.k3, self.k3), activation='relu', **self.kwargs)(y) 
        y = BatchNormalization()(y)
        y = Conv2D(K.int_shape(x)[-1], (self.k1, self.k1), **self.kwargs)(y)
        y = Add()([x,y]) 
        y = Activation('relu')(y)
        return y
    
    
    def FirstBlock(self,x,convF):
        x   = Conv2D(convF, (self.k9,self.k9), strides=self.s2, activation='relu', **self.kwargs)(x)
        x   = MaxPooling2D((self.k2, self.k2), strides=(self.s2, self.s2))(x) 
        for _ in range(2):
            x = BatchNormalization()(x)
            x = Conv2D(convF, (self.k3,self.k3), activation='relu', **self.kwargs)(x)
        return x
    
    
    def ConvBlockSubblock(self,x,convF,subbF):
        x = MaxPooling2D((self.k2, self.k2), strides=(self.s2, self.s2))(x) 
        x = BatchNormalization()(x)
        x = Conv2D(convF, (self.k1,self.k1), activation='relu', **self.kwargs)(x)
        for _ in range(4): x = self.Subblock(x, subbF)
        return x
    
    
    def PerFCNN(self,x):
        x = Conv2D(self.mid, (4, 1), activation='relu', padding='valid')(x)
        x = Reshape((self.BaseModel.output_shape[1], self.mid, 1))(x)
        x = Conv2D(1, (1, self.mid), activation='linear', padding='valid')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(1, use_bias=True, activation='sigmoid', name='weighted-average')(x)
        return x
        
    
    def SumF(self): return Lambda(lambda x: x[0]+x[1])
    def ProdF(self): return Lambda(lambda x: x[0]*x[1])
    def AbsDifF(self): return Lambda(lambda x: K.abs(x[0]-x[1]))
    def DifSqF(self): return Lambda(lambda x: K.square(x))
    
# =================================================================       


def ModelTraining(TIms,TLabs,VIms,VLabs,TeIms,TeLabs,path,Lr,lr,pm,n,shp,Epochs,compress_horizontally=True):
    print("Preprocessing {} images".format(len(os.listdir(path))))
    pre = ImgPreprocessing(path,shp,compress_horizontally)
    m = MYModel(shp,Lr)
    model = m.SiameseNet()
    
    Kcallback=[ModelCheckpoint(pm, monitor='val_loss', save_best_only=True,save_weights_only=True),
          ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr= lr, verbose=1)]
    
    print()
    print("==================================================")
    print()
    print("Training model has started")
    print()
    print()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print()
    print("Training images = ", len(TIms))
    print("Validation images = ", len(VIms))
    print("Test images = ", len(TeIms))
    print("Number of epochs = ", Epochs)
    print("Learning rate = ", Lr)
    print()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    genPair = ImsPairs(TIms, TLabs, batch_size=128)
    TImL, TImR, TY = Generator2Pairs(genPair)


    IsL,IsR,VY = GenTestPairs(VIms,VLabs,n)

    
    history = model.fit(BatchGenerator(TImL,TImR,TY,pre,shp,128),
                    validation_data=ValGenerator(IsL,IsR, VY,pre,shp,128),
                    epochs=Epochs,
                    callbacks=Kcallback,
                    max_queue_size=12,
                    workers=6)
    print()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print()
    print("Valloss, val acc = ", model.evaluate(ValGenerator(IsL,IsR, VY,pre,shp,128), verbose=0))
    print()
    print()
    print("+++++++++++++++++++++++++++++++++++++")
    print()
    print("computing embeddings of test images")
    print()
    print("+++++++++++++++++++++++++++++++++++++++")
    print()
    TeL,TeR,TeY = GenTestPairs(TeIms,TeLabs,n)
    print("Test loss, Test accuracy = ", model.evaluate(ValGenerator(TeL,TeR, TeY,pre,shp,128),verbose=0))
    print()
    print("+++++++++++++++++++++++++++++++++++++++")
    print()
    print("Training and evaluation has finished!")
    print()
    print("++++++++++++++++++++++++++++++++++++++")

