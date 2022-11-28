#Loading necessary libraries
# =============================
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D
from tensorflow.keras.layers import Input, GlobalMaxPooling2D,  MaxPooling2D, Dense
from tensorflow.keras.preprocessing.image import img_to_array,array_to_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from scipy.ndimage import affine_transform
from PIL import Image as pil_image
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import io
import os
import matplotlib.pyplot as plt

# =======================================================

class DataPreparation(object):
    
    def __init__(self,pcsv,out_nodes):
        self.pcsv = pcsv
        self.out_nodes = out_nodes
        
    def Im2side(self):
        df = pd.read_csv(self.pcsv)
        return dict(df.to_records(index=False))
    
    def Im2Lab1(self):
        Ims, Labs = list(zip(*[(img,lab) for img,lab in self.Im2side().items()]))
        Ims, Labs = np.array(Ims),np.array(Labs)
        Labels = np.where(Labs=="L",0.0,1.0)
        idx = np.arange(Ims.shape[0])
        np.random.shuffle(idx)
        return Ims[idx],Labels[idx]
    
    def Im2Lab2(self):
        Im2S = {}
        Im2s = self.Im2side()
        for img in Im2s.keys():
            if Im2s[img]=="L":Im2S[img]=0.0
            elif Im2s[img]=="R":Im2S[img]=1.0
            else:Im2S[img]=2.0
        Ims, Labs = list(zip(*[(img,lab) for img,lab in Im2S.items()]))
        Ims, Labs = np.array(Ims),np.array(Labs)
        idx = np.arange(Ims.shape[0])
        np.random.shuffle(idx)
        return Ims[idx],Labs[idx]
    
    def LabEncoding(self):
        """
        "L" is encoded as [1.0,0.0] and "R" as [0.0,1.0]
        """
        Ims, Labs = self.Im2Lab1() if self.out_nodes==2 else self.Im2Lab2()
        LabEnc = to_categorical(Labs)
        return Ims,LabEnc
    
# ===============================

class ImgPreprocessing(object):
    
    def __init__(self,path,shp):
        self.p = path
        self.shp = shp
        self.Ims2sz = self.Img2sz()
        
    def load_img(self,img):
        '''@param img: image to be read
        return a numpy array of the read image'''
        return img_to_array(pil_image.open(self.p+img).convert('L'))
    
    def norm(self,img):
        '''Normalisation to 0 mean and unity variance'''
        img -= np.mean(img,keepdims=True)
        img /= np.std(img,keepdims=True)+ K.epsilon()
        return img 
    
    def build_transform(self,rot,shea,h_zoom,w_zoom,h_shif,w_shif):
        '''Construct a transfo matrix with the specified characteristics'''
        r,s = np.deg2rad(rot),np.deg2rad(shea)
        rot_m,shi_m = self.rotat_m(r),self.shift_m(h_shif,w_shif)
        she_m,zoo_m = self.shea_m(s),self.zoom_m(h_zoom,w_zoom)
        shi_m      = self.shift_m(-h_shif,-w_shif)
        return np.dot(np.dot(rot_m, she_m), np.dot(zoo_m, shi_m))
    
    def bb_transform(self,img):
        size_x,size_y = self.Ims2sz[img]
        x0,y0,x1,y1 = 0,0,size_x,size_y
        big = max(x1,y1)
        sx = big - x1
        sy = big - y1
        l,r = sx//2,sx//2
        b,t = sy//2,sy//2
        # padd a half on the left and on the right
        x0 -= l
        x1 += r
        # pad a half on bottom and on top
        y0 -= b
        y1 += t  
        return x0,y0,x1,y1
    
    def read_crop(self,img,augment):
        '''@param img: image to be transformed
        @param augment:True/False if data augmentation should be applied
        return a numpy array with the transformed image
        '''

        x0,y0,x1,y1 = self.bb_transform(img)
        trans = self.minus()
        trans = np.dot(self.scale(x0,y0,x1,y1),trans)
        if augment:
            trans = np.dot(self.build_transform(random.uniform(-5, 5),random.uniform(-2, 2),
                random.uniform(0.8, 1.0),random.uniform(0.8, 1.0),random.uniform(-0.05*(y1 - y0),0.05*(y1 - y0)),
                random.uniform(-0.05*(x1 - x0), 0.05*(x1 - x0))), trans)
        trans = np.dot(self.plus(x0,y0,x1,y1), trans)
        img = self.load_img(img)
        img = self.transfo_img(img,trans)
        return self.norm(img)

    def transfo_img(self,img, affine):
        '''@param img: image to which a transfo is applied
        @param affine: an affine transformation to be applied
        return a transformed image'''
        matrix,offset   = affine[:2,:2],affine[:2,2]
        img = img.reshape(img.shape[:-1])
        img = affine_transform(img, matrix, offset, output_shape=self.shp[:-1], order=1,
                                     mode='constant', cval=np.average(img))
        return img.reshape(self.shp)
    
 
    def read_for_training(self,p):
        '''read and preprocess an image with data augmentation (random transfo)'''
        return self.read_crop(p,True)
    
    def read_for_validation(self,p):
        '''read and preprocess an image without data augmentation (testing phase)'''
        return self.read_crop(p,False)
    
    # implementation of different transformations
    def minus(self): return np.array([[1, 0, -0.5*self.shp[0]], [0, 1, -0.5*self.shp[1]], [0, 0, 1]])
    def plus(self,x0,y0,x1,y1):return np.array([[1, 0, 0.5*(y1 + y0)], [0, 1, 0.5*(x1 + x0)], [0, 0, 1]])
    def scale(self,x0,y0,x1,y1): return np.array([[(y1 - y0)/self.shp[0], 0, 0], [0, (x1 - x0)/self.shp[1], 0], [0, 0, 1]])      
    def rotat_m(self,a): return np.array([[np.cos(a),np.sin(a),0],[-np.sin(a),np.cos(a),0],[0,0,1]])
    def shift_m(self,h,w): return np.array([[1, 0, h], [0, 1, w], [0, 0, 1]])
    def zoom_m(self,h,w): return np.array([[1.0/h, 0, 0], [0, 1.0/w, 0], [0, 0, 1]])
    def shea_m(self,a): return np.array([[1, np.sin(a), 0], [0, np.cos(a), 0], [0, 0, 1]])
    
    def img_sz(self,img):return pil_image.open(self.p+img).size
    def Img2sz(self):  return {img:self.img_sz(img) for img in tqdm(os.listdir(self.p))}
    
# ====================================================

class BatchGenerator(Sequence):

    
    def __init__(self, Ims, Labels,pre,shp, batch_size=128):
        self.pre = pre
        self.Ims       = Ims
        self.Labels    = Labels
        self.shp = shp
        self.batch_size = batch_size
        super().__init__()
          
    def __getitem__(self, index):
        start = self.batch_size*index
        size  = min(len(self.Ims) - start, self.batch_size)
        ImsA  = np.zeros((size,) + self.shp, dtype=K.floatx())
        ImsL  = np.zeros((size,) + self.Labels[0].shape, dtype = K.floatx())
        for i in range(size):
            ImsA[i,:,:,:] = self.pre.read_for_training(self.Ims[start + i])
            ImsL[i,:]     = self.Labels[start+i]
        indexes = np.arange(size)
        np.random.shuffle(indexes)
        ImsA = ImsA[indexes]
        ImsL = ImsL[indexes]
        
        return ImsA,ImsL
    
    def __len__(self):
        return (len(self.Ims) + self.batch_size - 1)//self.batch_size

# ========================================

class ValGenerator(Sequence):

    
    def __init__(self, Ims, Labels,pre,shp,batch_size=64):
        self.pre = pre
        self.Ims       = Ims
        self.Labels    = Labels
        self.shp = shp
        self.batch_size = batch_size
        super().__init__()
          
    def __getitem__(self, index):
        start = self.batch_size*index
        size  = min(len(self.Ims) - start, self.batch_size)
        ImsA  = np.zeros((size,) + self.shp, dtype=K.floatx())
        ImsL  = np.zeros((size,) + self.Labels[0].shape, dtype = K.floatx())
        for i in range(size):
            ImsA[i,:,:,:] = self.pre.read_for_validation(self.Ims[start + i])
            ImsL[i,:]     = self.Labels[start+i]
        indexes = np.arange(size)
        np.random.shuffle(indexes)
        ImsA = ImsA[indexes]
        ImsL = ImsL[indexes]
        
        return (ImsA,ImsL)
    
    def __len__(self):
        return (len(self.Ims) + self.batch_size - 1)//self.batch_size
    
# ===================================================


class ModelArchitecture(object):
    
    def __init__(self,shp,out_nodes,l2=0.0, k1=1, k2=2, k3=3, k9=9, mid =32):
        
        self.k1,self.k2,self.k3,self.k9,self.mid = k1,k2,k3,k9,mid
        self.regul = regularizers.l2(l2)
        self.kwargs = {'padding':'same', 'kernel_regularizer':self.regul}
        self.s2 = self.k2
        self.shp = shp
        self.out_nodes = out_nodes
          
    def OrientationModel(self):
        Ims = Input(shape=self.shp)
        x = self.FirstBlock(Ims,64)
        x = self.ConvBlockSubblock(x,128,64)
        x = self.ConvBlockSubblock(x,256,64)
        x = self.ConvBlockSubblock(x,384,96)
        x = self.ConvBlockSubblock(x,512,128)
        x = GlobalMaxPooling2D()(x)
        x = Dense(256,activation='relu')(x)
        x = Dense(128,activation='relu')(x)
        x = Dense(self.out_nodes,activation = 'softmax')(x)
        return Model(Ims,x,name='orientation_model')
    
    
    def Subblock(self,x, convF):
        x = BatchNormalization()(x)
        y = x
        y = Conv2D(convF, (self.k1, self.k1), activation='relu', **self.kwargs)(y) #reduce the nr of feature to filter
        y = BatchNormalization()(y)
        y = Conv2D(convF, (self.k3, self.k3), activation='relu', **self.kwargs)(y) # extend the feature field
        y = BatchNormalization()(y)
        y = Conv2D(K.int_shape(x)[-1], (self.k1, self.k1), **self.kwargs)(y)# restore the nr of original features
        y = Add()([x,y]) # add a skip connection
        y = Activation('relu')(y)
        return y
    
    
    def FirstBlock(self,x,convF):
        x   = Conv2D(convF, (self.k9,self.k9), strides=self.s2, activation='relu', **self.kwargs)(x)
        x   = MaxPooling2D((self.k2, self.k2),padding ="same",  strides=(self.s2, self.s2))(x) 
        for _ in range(2):
            x = BatchNormalization()(x)
            x = Conv2D(convF, (self.k3,self.k3), activation='relu', **self.kwargs)(x)
        return x
    
    
    def ConvBlockSubblock(self,x,convF,subbF):
        x = MaxPooling2D((self.k2, self.k2),padding ="same", strides=(self.s2, self.s2))(x) 
        x = BatchNormalization()(x)
        x = Conv2D(convF, (self.k1,self.k1), activation='relu', **self.kwargs)(x)
        for _ in range(4): x = self.Subblock(x, subbF)
        return x
    
    
# =================================================

def ModelTraining(TIms,TLabs,VIms,VLabs,TeIms,TeLabs,path,Lr,pw,shp,Epochs,out_nodes):
    print("Preprocessing {} images".format(len(os.listdir(path))))
    pre = ImgPreprocessing(path,shp)
    m = ModelArchitecture(shp,out_nodes)
    model = m.OrientationModel()
    Kcallback=[ModelCheckpoint(pw, monitor='val_loss', save_best_only=True,save_weights_only=True)]

    model.compile(loss="categorical_crossentropy",metrics=["accuracy"], optimizer=Adam(1e-4))

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

    
    
    history = model.fit(BatchGenerator(TIms, TLabs,pre, shp,128),
                    validation_data=ValGenerator(VIms, VLabs,pre,shp,128),
                    epochs=Epochs,
                    callbacks=Kcallback,
                    max_queue_size=12,
                    workers=6)
    print()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print()
    model.load_weights(pw)
    print("Val loss, Val accuracy = ", model.evaluate(ValGenerator(VIms, VLabs,pre,shp,128), verbose=0))
    print()
    print()
    print("+++++++++++++++++++++++++++++++++++++")
    print()
    print()
    print("Test loss, test accuracy = ", model.evaluate(ValGenerator(TeIms, TeLabs,pre,shp,128), verbose=0))
    print()
    print()
    print("+++++++++++++++++++++++++++++++++++++")
    print()
    print()
    print("Training has finished")
    print()
    print("")






