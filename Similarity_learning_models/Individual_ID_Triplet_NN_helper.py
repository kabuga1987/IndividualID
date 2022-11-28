# Importing necessary libraries
#  ===============================
from tensorflow.keras.layers import Activation,Add,BatchNormalization,Conv2D
from tensorflow.keras.layers import Input,GlobalMaxPooling2D,MaxPooling2D
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
#import matplotlib.pyplot as plt

# ======================================================

class DataStructuring(object):
    
    
    def __init__(self,p1,p2,nIms):
        self.p1 = p1
        self.p2 = p2
        self.nIms = nIms
        self.ID2Imgs = self.ID2Ims()
        
    
    def Img2ID(self):
        Ims = os.listdir(self.p2)
        df = pd.read_csv(self.p1)
        df  = df[df.Image.isin(Ims)]
        return dict([(img,ID) for img,ID in df.to_records(index=False)])
    
    
    def ID2Ims(self):
        ID2Ims = {}
        for Img,ID in self.Img2ID().items():
            if ID not in ID2Ims:
                ID2Ims[ID]= [Img]
            else:
                ID2Ims[ID].append(Img)
        return ID2Ims
    
    def TrainDataStructure(self):
        lstoflst = list(self.ID2Imgs.values())
        mx = max([len(lst) for lst in lstoflst])
        LstofAllIms = np.zeros((len(lstoflst), mx),dtype='<U64')
        for i,lst in enumerate(lstoflst):
            lengthtopad = mx-len(lst)
            zeros = np.zeros(lengthtopad,)
            padwithzeros = np.hstack([lst,zeros])
            LstofAllIms[i] = padwithzeros
            idx = np.arange(LstofAllIms.shape[0])
        return LstofAllIms[idx]
    
    def LstofAllIms(self):
        L=[]
        x = self.TrainDataStructure()
        N = x.shape[1]
        mults = [n for n in range(N) if n%self.nIms==0]
        for i in mults:
            lst = x[:,i:i+self.nIms].flatten().tolist()
            L += lst
        LA = np.array(L)
        return LA[LA!='0.0']
    
    
    def Im2Label(self):
        AllIms = self.LstofAllIms()
        Im2Labs = dict([(a[j],i)for i,a in enumerate(self.ID2Imgs.values()) for j in range(len(a))])
        return zip(*[(img,Im2Labs[img]) for img in AllIms])
    
    
# ================================================================

class Augmentation():
    
    def __init__(self, TIs,TLs,n_augm,nIms):
        self.TIs, self.TLs = TIs, TLs
        self.n_augm = n_augm
        self.nIms = nIms
        
        
    def ID2Ims(self):
        ID2Ims = {}
        for Img,ID in zip(self.TIs,self.TLs):
            if ID not in ID2Ims:
                ID2Ims[ID]= [Img]
            else:
                ID2Ims[ID].append(Img)
        return ID2Ims
    
    def TrainDataStructure(self):
        lstoflst = list(self.ID2Ims().values())
        mx = max([len(lst) for lst in lstoflst])
        LstofAllIms = np.zeros((len(lstoflst), mx),dtype='<U64')
        for i,lst in enumerate(lstoflst):
            lengthtopad = mx-len(lst)
            zeros = np.zeros(lengthtopad,)
            padwithzeros = np.hstack([lst,zeros])
            LstofAllIms[i] = padwithzeros
        return LstofAllIms
    
    
    def combination(self):
        matx = self.TrainDataStructure()
        IdX = []
        for i in range(self.n_augm):
            idx = np.arange(matx.shape[0])
            np.random.shuffle(idx)
            IdX = IdX+[idx]
        return np.vstack([matx[IdX[i]]for i in range(len(IdX))])
    
    def LstofAllIms(self):
        xx = self.combination()
        L=[]
        N = xx.shape[1]
        mults = [n for n in range(N) if n%self.nIms==0]
        for i in mults:
            lst = xx[:,i:i+self.nIms].flatten().tolist()
            L += lst
        LA = np.array(L)
        return LA[LA!='0.0']
    
    def Im2Label(self):
        AllIms = self.LstofAllIms()
        Im2Labs = dict([(a[j],i)for i,a in enumerate(self.ID2Ims().values()) for j in range(len(a))])
        return zip(*[(img,Im2Labs[img]) for img in AllIms])
            
# ==================================================================


class ImgPreprocessing(object):
    
    def __init__(self,path,shp,compress_horizontally=False):
        self.p = path
        self.cph = compress_horizontally
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
       
    def bb_transform1(self,img):
        size_x,size_y = self.Ims2sz[img]
        x0,y0,x1,y1   = 0,0,size_x,size_y
        dx,dy           = x1 - x0,y1 - y0
        ratio = 2.0 # only used for whales
        if dx > dy*ratio:
            dy  = 0.5*(dx/ratio - dy)
            y0 -= dy
            y1 += dy
        else:
            dx  = 0.5*(dy*ratio - dx)
            x0 -= dx
            x1 += dx
        return x0,y0,x1,y1
    
    def bb_transform2(self,img):
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

        x0,y0,x1,y1 = self.bb_transform1(img) if self.cph else self.bb_transform2(img)
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
    

# =============================================================


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
        ImsL  = np.zeros((size,), dtype = K.floatx())
        for i in range(size):
            ImsA[i,:,:,:] = self.pre.read_for_training(self.Ims[start + i])
            ImsL[i]     = self.Labels[start+i]
        indexes = np.arange(size)
        np.random.shuffle(indexes)
        ImsA = ImsA[indexes]
        ImsL = ImsL[indexes]
        
        return ImsA,ImsL
    
    def __len__(self):
        return (len(self.Ims) + self.batch_size - 1)//self.batch_size


# ==================================================================


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
        ImsL  = np.zeros((size,), dtype = K.floatx())
        for i in range(size):
            ImsA[i,:,:,:] = self.pre.read_for_validation(self.Ims[start + i])
            ImsL[i]     = self.Labels[start+i]
        indexes = np.arange(size)
        np.random.shuffle(indexes)
        ImsA = ImsA[indexes]
        ImsL = ImsL[indexes]
        
        return (ImsA,ImsL)
    
    def __len__(self):
        return (len(self.Ims) + self.batch_size - 1)//self.batch_size


# ====================================================================


class TestGenerator(Sequence):

    
    def __init__(self, TeIs,pre,shp, batch_size=64):
        self.TeIs = TeIs
        self.pre = pre
        self.shp = shp
        self.batch_size = batch_size
        super().__init__()
          
    def __getitem__(self, index):
        start = self.batch_size*index
        size  = min(len(self.TeIs) - start, self.batch_size)
        ImsA  = np.zeros((size,) + self.shp, dtype=K.floatx())
        for i in range(size):
            ImsA[i,:,:,:] = self.pre.read_for_validation(self.TeIs[start + i])
        return ImsA
    
    def __len__(self):
        return (len(self.TeIs) + self.batch_size - 1)//self.batch_size


# ===========================================


class ModelArchitecture(object):
    
    def __init__(self,shp,l2=0.0, k1=1, k2=2, k3=3, k9=9, mid =32):
        
        self.k1,self.k2,self.k3,self.k9,self.mid = k1,k2,k3,k9,mid
        self.regul = regularizers.l2(l2)
        self.kwargs = {'padding':'same', 'kernel_regularizer':self.regul}
        self.s2 = self.k2
        self.shp = shp
          
    def EmbeddingModel(self):
        Ims = Input(shape=self.shp)
        x = self.FirstBlock(Ims,64)
        x = self.ConvBlockSubblock(x,128,64)
        x = self.ConvBlockSubblock(x,256,64)
        x = self.ConvBlockSubblock(x,384,96)
        x = self.ConvBlockSubblock(x,512,128)
        Embs = GlobalMaxPooling2D()(x)
        Embmodel = Model(Ims,Embs)
        return Embmodel
    
    
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

# ========================================================================

class Predictions(object):
    """Compute the evaluation metrics and the optimized threshold
    Normally the threshold is optimized on validation set and then used
    for the test set."""
    
    def __init__(self,Lembs,Rembs,Y):
        self.Lembs = Lembs
        self.Rembs = Rembs
        self.Y = Y
        #=====================================
        self.distances = self.compute_dist()
        self.fpr,self.tpr,self.thresholds, self.auc= self.compute_metrics()
        #================================
        
        self.prec, self.recal, self.thr = self.prec_recal_thr()
        _,_,_,self.thresholdopt = self.f1score()
        
    def compute_dist(self):
        """
        inputs
        a,b : Embeddings, tensors of shape (n,m); n nr of embs and m dim of eac emb
        returns
            vector of arrays that are distances between image pairs
        """
        return np.linalg.norm(self.Lembs-self.Rembs,axis=1)
    
    def compute_metrics(self):
        # calculate AUC
        auc = roc_auc_score(self.Y, -self.distances)
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(self.Y, -self.distances)
        return fpr, tpr, thresholds,auc
    

    
    def calc_acc(self):
        pres = np.where(self.distances<= np.abs(self.thresholdopt),1,0)
        return np.where(self.Y==pres)[0].shape[0]/self.Y.shape[0]
    
           
    def prec_recal_thr(self):
        return precision_recall_curve(self.Y, -self.distances)
    
    def f1score(self):
        f1s =  2*self.prec*self.recal/(self.prec+self.recal)
        idx = np.argmax(f1s)
        thrOpt = round(np.abs(self.thr)[idx],ndigits=4)
        precOpt = round(self.prec[idx],ndigits=4)
        recalOpt = round(self.recal[idx],ndigits=4)
        f1sOpt = round(f1s[idx],ndigits=4)
        return precOpt,recalOpt,f1sOpt,thrOpt
    
    # ========================================================================
    
    
    
def Select_val_pairs(IDs,n):
    """"select an equal number of matches and non-matches from precomputed 
    embeddings;these are used to evaluation the similarity learning network
    In total we have 2n pairs, n matches and n non-matches.
    """
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
    #return lef-and right-side indexes for selecting corresponding embeddings and their labels L[idx]
    return XL[Idx],XR[Idx],L[Idx]



def ModelTraining(TIms,TLabs,VIms,VLabs,TeIms,TeLabs,path,Lr,lr,pm,n,shp,Epochs,compress_horizontally=True):
    print("Preprocessing {} images".format(len(os.listdir(path))))
    pre = ImgPreprocessing(path,shp,compress_horizontally)
    m = ModelArchitecture(shp)
    model = m.EmbeddingModel()
    model.compile(loss=tfa.losses.TripletSemiHardLoss(), optimizer=Adam(Lr))
    
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
    print("Learning rate upper limit = ", Lr)
    print("Learning rate lower limit = ", lr)
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
#     print()
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title(' Training and validation loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epochs')
#     plt.legend(['Train', 'val'], loc='upper right')
#     plt.grid()
#     plt.savefig(pm+'test.pdf')
#     plt.clf()
#     model.load_weights(pm+"test.h5")
#     print()
    print("Val loss = ", model.evaluate(ValGenerator(VIms, VLabs,pre,shp,128), verbose=0))
    print()
    print()
    print("+++++++++++++++++++++++++++++++++++++")
    print()
    print("computing embeddings of test images")
    print()
    Embs = model.predict(TestGenerator(TeIms,pre,shp, 128),verbose=1)
    Lidx,Ridx,Y = Select_val_pairs(TeLabs,n)
    Lembs, Rembs = Embs[Lidx], Embs[Ridx]
    print()
    print("+++++++++++++++++++++++++++++++++++++")
    print()
    print("computing Evaluation Metrics")
    print()
    print("+++++++++++++++++++++++++++++++++++++")
    print()
    p = Predictions(Lembs,Rembs,Y)
    precision, recall, f1score, threshold = p.f1score()
    auc = p.auc
    accuracy = p.calc_acc()
    print("Number of test pairs = ", Y.shape[0])
    print()
    print("Recall = ", recall)
    print()
    print("Precision = ", precision)
    print()
    print("F1 score = ", f1score)
    print()
    print("Accuracy = ", accuracy)
    print()
    print("AUC = ", auc)
    print()
    print("Optimized threshold = ", threshold)
    print("++++++++++++++++++++++++++++++++++++++")
    print()
    print("Training and evaluation has finished!")
    print()
    print("======================================")

