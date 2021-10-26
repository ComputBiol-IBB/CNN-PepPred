import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import numpy.matlib
import sklearn
import sklearn.metrics
import math
import pandas as pd
import tensorflow as tf
from keras import initializers, optimizers
from keras.regularizers import l2
from keras.models import Sequential,load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import time
import logomaker as lm
import pickle
from pathlib import Path
from datetime import datetime

class CNNPepPred:
    def __init__(self,allele='no_allele_name',savePath=Path(os.getcwd()),doTraining=False,trainingData=None,trainingOutcome=None,doLogoSeq=False,doCV=False,cvPart=None,kFold=5,doApplyData=False,trainedModelsFile=None,applyData=None,applyDataName=None,epitopesLength=15,parametersFile='parameters.txt'):
        self.__version__ = '0.1.1'
        self.allele = allele.translate ({ord(c): "_" for c in "!@#$%^&*()[]{};:,./<>?\|`~-= +"})
        savePath = Path(savePath)
        if not os.path.isdir(savePath):
            os.mkdir(savePath)
        self.savePath = savePath / (allele+datetime.now().strftime("_%m#%d#%Y#%H#%M#%S"))
        os.mkdir(self.savePath)
        self.doTraining = doTraining
        self.epitopesLength = epitopesLength
        self.doApplyData = doApplyData
        self.doLogoSeq = doLogoSeq
        if parametersFile!=None:
            parametersFile = Path(parametersFile)
            if os.path.exists(parametersFile):
                self.parameters = pd.read_csv(parametersFile,index_col='Parameter_Name')
                self.getParameters()
        if doTraining:
            if cvPart is None:
                trainingData,trainingOutcome = sklearn.utils.shuffle(trainingData,trainingOutcome)
            self.trainingData = trainingData
            self.trainingOutcome = trainingOutcome
            self.doCV = doCV
            if doCV:
                if cvPart is None:
                    self.generateCVpartWithLeastLmerOverlap(kFold,saveCVPart=True)
                else:
                    self.cvPart = cvPart
            if doApplyData:
                self.applyData = applyData
                self.applyDataName = applyDataName
        if (not doTraining and (doApplyData or doLogoSeq)) or (trainedModelsFile is not None):
            if trainedModelsFile==None:
                trainedModelsFile = Path(os.getcwd()) / 'trainedIEDBmodels' / ('model_'+self.allele)
            trainedModelsFile = Path(trainedModelsFile)
            isNotFolder = True
            if not trainedModelsFile.name.endswith('.pkl'):
                d = os.listdir(trainedModelsFile)
                ind_pkl = np.where([f.endswith('.pkl') for f in d])[0]
                if ind_pkl.size==0:
                    netspath = trainedModelsFile / ('nets')
                    self.parameters = pd.read_csv(trainedModelsFile / 'parameters.txt',index_col='Parameter_Name')
                    trainingDataModelFile = trainedModelsFile / "".join((trainedModelsFile.name.replace('model_','data_'),'.txt'))
                    if os.path.exists(trainingDataModelFile):
                        self.trainingDataModel = pd.read_csv(trainingDataModelFile,header=None).iloc[0:None,0].to_list()
                    valueColName = "Parameter_Value"
                    if isinstance(self.parameters[valueColName]['maxL'],pd.Series):
                        self.maxL = int(self.parameters[valueColName]['maxL'].iloc[0]) #careful with repeated maxL in param file
                    else:
                        self.maxL = int(self.parameters[valueColName]['maxL'])
                    isNotFolder = False
                    parentFolder = trainedModelsFile
                else:
                    trainedModelsFile = trainedModelsFile / d[ind_pkl[0]]
            if trainedModelsFile.name.endswith('.pkl'):
                trainedModelCNN = self.load_object(Path(trainedModelsFile))
                self.seqL = trainedModelCNN.seqL
                self.maxL = trainedModelCNN.maxL
                self.nMaxPool = trainedModelCNN.nMaxPool
                netspath = trainedModelCNN.savePath / ('nets')
                self.timeTrain = trainedModelCNN.timeTrain
                self.parameters = trainedModelCNN.parameters
                parentFolder = trainedModelCNN.savePath
                trainingDataModelFile = trainedModelsFile.savePath / "".join((trainedModelsFile.savePath.name.replace('model_','data_'),'.txt'))
                if os.path.exists(trainingDataModelFile):
                    self.trainingDataModel = pd.read_csv(trainingDataModelFile,header=None).iloc[0:None,0].to_list()
            elif isNotFolder:
                 raise NameError("The given trained model does not have the correct format. It must either be .pkl saved model or a folder containing a parameter file and a folder with the trained nets.")
            if parametersFile!=None:
                valueColName = "Parameter_Value"
                self.parameters[valueColName][['nbRandSeq','nbBest','lengthRandSeq','maxNbSamples2apply']] = [self.nbRandSeq,self.nbBest,self.lengthRandSeq,self.maxNbSamples2apply]
                if doTraining:
                    str_interval = lambda x: str(x).replace('[','').replace(']','').replace(', ','/')
                    self.parameters[valueColName][['alpha','gamma','maxEpochs','miniBatchSize','activationFctDenseLayer','lossFct','initializeStd']] = [str_interval(self.alpha),str_interval(self.gamma),str_interval(self.maxEpochs),str_interval(self.miniBatchSize),self.activationFctDenseLayer,self.lossFct,self.initializeStd]
            self.getParameters()
            if not os.path.isdir(netspath):
                raise NameError('There is no folder called "nets" with the trained nets in the location of the given model: \n %s' % parentFolder)
            self.trainedModels = [load_model(netspath / nn) for nn in os.listdir(netspath)]
            if doApplyData:
                self.applyData = applyData
                self.applyDataName = applyDataName
                if self.epitopesLength!=None and self.epitopesLength>self.maxL:
                    raise NameError('The length of the epitopes (%i) is bigger than the maximal length of the model (%i)' %(self.epitopesLength,self.maxL))
            
    def getParameters(self):
        valueColName = "Parameter_Value"
        params = self.parameters
        bindingThr = params[valueColName]["bindingThr"]
        if bindingThr==bindingThr:
            bindingThr = float(bindingThr)
        else:
            bindingThr = 0.5
        self.bindingThr = bindingThr
        similarityMat = params[valueColName]["similarityMat"]
        if similarityMat!=similarityMat:
            similarityMat = 'blosum62.txt'
        self.S = pd.read_csv(Path(similarityMat),index_col=0).to_numpy(dtype=float)
        l = params[valueColName]["l"]
        if l==l:
            l = int(l)
        else:
            l = 9
        self.l = l
        maxNbSamples2apply = params[valueColName]["maxNbSamples2apply"]
        if maxNbSamples2apply==maxNbSamples2apply:
            maxNbSamples2apply = int(maxNbSamples2apply)
        else:
            maxNbSamples2apply = 50000
        self.maxNbSamples2apply = maxNbSamples2apply
        nbPrev = params[valueColName]["nbPrev"]
        if nbPrev==nbPrev:
            nbPrev = int(nbPrev)
        else:
            nbPrev = 2
        self.nbPrev = nbPrev
        nbAfter = params[valueColName]["nbAfter"]
        if nbAfter==nbAfter:
            nbAfter = int(nbAfter)
        else:
            nbAfter = 2
        self.nbAfter = nbAfter
        F = params[valueColName]["F"]
        if F==F:
            F=[int(x) for x in F.split('/')]
        else:
            F = [5,10,20,30]
        self.F = F
        rep = params[valueColName]["rep"]
        if rep==rep:
            rep = int(rep)
        else:
            rep = 10
        self.rep = rep
        nMaxPool = params[valueColName]["nMaxPool"]
        if nMaxPool==nMaxPool:
            nMaxPool = int(nMaxPool)
        else:
            nMaxPool = None
        self.nMaxPool = nMaxPool
        initializeStd = params[valueColName]["initializeStd"]
        if initializeStd==initializeStd:
            initializeStd = float(initializeStd)
        else:
            initializeStd = 0.01
        self.initializeStd = initializeStd
        alpha = params[valueColName]["alpha"]
        if alpha==alpha:
            alpha = [float(x) for x in alpha.split('/')]
        else:
            alpha = [0.005]
        self.alpha = alpha
        gamma = params[valueColName]["gamma"]
        if gamma==gamma:
            gamma = [float(x) for x in gamma.split('/')]
        else:
            gamma = [0.9]
        self.gamma = gamma
        l2_fact = params[valueColName]["l2_fact"]
        if l2_fact==l2_fact:
            l2_fact = float(l2_fact)
        else:
            l2_fact = 0.0001
        self.l2_fact = l2_fact
        maxEpochs = params[valueColName]["maxEpochs"]
        if maxEpochs==maxEpochs:
            maxEpochs = [int(x) for x in maxEpochs.split('/')]
        else:
            maxEpochs = [30]
        self.maxEpochs = maxEpochs
        miniBatchSize = params[valueColName]["miniBatchSize"]
        if miniBatchSize==miniBatchSize:
            miniBatchSize = [int(x) for x in miniBatchSize.split('/')]
        else:
            miniBatchSize = [128]
        self.miniBatchSize = miniBatchSize
        useBias = params[valueColName]["useBias"]
        if useBias==useBias:
            useBias = int(useBias)>0
        else:
            useBias = True
        self.useBias = useBias
        activationFctDenseLayer = params[valueColName]["activationFctDenseLayer"]
        if activationFctDenseLayer!=activationFctDenseLayer:
            activationFctDenseLayer = 'linear'
        self.activationFctDenseLayer = activationFctDenseLayer
        lossFct = params[valueColName]["lossFct"]
        if lossFct!=lossFct:
            lossFct = 'mean_squared_error'
        self.lossFct = lossFct
        nbRandSeq = params[valueColName]["nbRandSeq"]
        if nbRandSeq==nbRandSeq:
            nbRandSeq = int(nbRandSeq)
        else:
            nbRandSeq = 200000
        self.nbRandSeq = nbRandSeq
        nbBest = params[valueColName]["nbBest"]
        if nbBest==nbBest:
            nbBest = int(nbBest)
        else:
            nbBest = 2000
        self.nbBest = nbBest
        lengthRandSeq = params[valueColName]["lengthRandSeq"]
        if lengthRandSeq==lengthRandSeq:
            lengthRandSeq = int(lengthRandSeq)
        else:
            lengthRandSeq = 15
        self.lengthRandSeq = lengthRandSeq
        
        
    def aa2int(self,s):
        if isinstance(s,str):
            s = [s]
        N = len(s)
        sInt = []
        mapp= np.array([1,0,5,4,7,14,8,9,10,0,12,11,13,3,0,15,6,2,16,17,0,20,18,0,19,0,21],dtype=int)-1
        mappAA='ARNDCQEGHILKMFPSTWYV'
        validInt = np.array([ord(letter) - 96 for letter in mappAA.lower()],dtype=int)-1
        for i in range(0,N):
            numbers = np.array([ord(letter) - 96 for letter in s[i].lower()])-1
            numbers[np.all([numbers!=v for v in validInt],axis=0)] = 26
            sInt.append(mapp[numbers])
        return sInt
    
    def int2aa(self,sInt):
        if not isinstance(sInt,list) and sInt.ndim==1:
            sInt = [sInt]
        N = len(sInt)
        mapp='ARNDCQEGHILKMFPSTWYV-'
        s = ["".join([mapp[sInt[i][pos]] for pos in np.arange(0,sInt[i].shape[0])]) for i in np.arange(0,N)]
        return s
    
    def seqLength(self,s,saveOutput=False):
        l = self.l
        N = len(s)
        seqL = np.zeros((N,1),dtype=int)
        for i in range(0,N):
            seqL[i] = len(s[i])
        maxL = np.max(seqL)
        nMaxPool = self.nMaxPool
        if nMaxPool == None:
            l_most_freq = np.argmax(np.histogram(seqL,range(1,maxL+2))[0])+1
            nMaxPool =np.max([6,maxL-l-l_most_freq+self.nbPrev+self.nbAfter+2])
        if saveOutput:
            self.seqL = seqL
            if not hasattr(self,'maxL'):
                self.maxL = maxL
                self.nMaxPool = nMaxPool
            elif self.maxL<maxL:
                print('The class already has an attribute "maxL" and it is smaller than the new value to assign. Consequently, if further training is performed, some data with larger length might be ignored.')
        return seqL,maxL,nMaxPool
     
    def addEmptyPositions(self,sInt):
        seqL = self.seqLength(sInt)
        maxL = self.maxL
        nbPrev = self.nbPrev
        nbAfter = self.nbAfter
        seqL = seqL[0]
        if np.any(seqL>maxL):
            ind2keep = (seqL<=maxL).nonzero()[0]
            n_rmv = np.sum(seqL>maxL)
            seqL = seqL[ind2keep]
            if hasattr(self,'trainingData') and len(self.trainingData)==len(sInt):
                self.trainingData = [self.trainingData[i] for i in ind2keep]
                if hasattr(self,'trainingOutcome'):
                    self.trainingOutcome = self.trainingOutcome[ind2keep]
                if hasattr(self,'cvPart'):
                    self.cvPart = self.cvPart[ind2keep]
            print('%i instances (out of %i) with length larger than the maximal length %i have been ignored.' % (n_rmv,len(sInt),maxL))
            sInt = [sInt[i] for i in ind2keep]
        N = len(sInt)
        L = maxL+nbAfter
        intPrev =20*np.ones((1,nbPrev),dtype=int)[0]
        sIntNew = []
        for i in range(0,N):
            intAfter =  20*np.ones((1,L-seqL[i][0]),dtype=int)[0]
            sIntNew.append(np.concatenate((intPrev,sInt[i],intAfter)))
        return sIntNew
    
    def getImages(self,sInt):
        sInt = np.array(sInt)
        S = self.S
        N = sInt.shape[0]
        h = sInt[0].shape[0]
        w = S.shape[0]
        c = 1
        IM = np.zeros((N,h,w,c))
        for i in np.arange(0,N):
            IM[i,:,:,0] = S[sInt[i],0:w]
        IM=np.float32(IM)
        return IM
    
    def trainCNN(self,IM,out,saveModel=False):
        time_start = time.perf_counter()
        nMaxPool = self.nMaxPool
        IM,out=sklearn.utils.shuffle(IM,out)
        F = self.F
        rep = self.rep
        F = np.matlib.repmat(F,1,rep)[0]
        models = []
        nbRep = F.shape[0]
        N,h,w,c = IM.shape
        l=self.l
        initializeStd = self.initializeStd
        alpha = self.alpha
        gamma = self.gamma
        l2_fact = self.l2_fact
        maxEpochs = self.maxEpochs
        miniBatchSize = self.miniBatchSize
        useBias = self.useBias
        lossFct = self.lossFct
        actFct = self.activationFctDenseLayer
        do_transfer_learning = hasattr(self,'trainedModels')
        for i in np.arange(0,nbRep):
            model = Sequential()
            if do_transfer_learning:
                layer_conv = self.trainedModels[i].layers[0]
                layer_conv.trainable = False
                model.add(layer_conv)
            else:
                model.add(Conv2D(F[i], kernel_size=(l,w),strides=(1,w), activation='relu',kernel_regularizer=l2(l2_fact),kernel_initializer=initializers.RandomNormal(stddev=initializeStd),bias_initializer=initializers.Zeros(),input_shape=(h,w,c),use_bias=useBias))
            model.add(MaxPooling2D(pool_size=(nMaxPool,1),strides=(1,1)))
            model.add(Flatten())
            model.add(Dense(1,activation=actFct,kernel_initializer=initializers.RandomNormal(stddev=initializeStd),bias_initializer=initializers.Zeros(),kernel_regularizer=l2(l2_fact),use_bias=useBias))
            opt = optimizers.sgd(learning_rate=alpha[0],momentum=gamma[0])
            model.compile(optimizer=opt, loss=lossFct)
            model.fit(IM, out, epochs=maxEpochs[0],batch_size=miniBatchSize[0],verbose=0,shuffle=1,workers=1)
            if do_transfer_learning and maxEpochs[-1]>0:
                layer_conv.trainable = True
                opt = optimizers.sgd(learning_rate=alpha[-1],momentum=gamma[-1])
                model.compile(optimizer=opt, loss=lossFct)
                model.fit(IM, out, epochs=maxEpochs[-1],batch_size=miniBatchSize[-1],verbose=0,shuffle=1,workers=1)
            models.append(model)
        time_elapsed = (time.perf_counter() - time_start)
        if saveModel:
            self.trainedModels = models
            self.timeTrain = time_elapsed
            modelPath = self.savePath / ('model_'+self.allele)
            os.mkdir(modelPath)
            netPath = modelPath / ('nets')
            os.mkdir(netPath)
            for i,m in enumerate(models):
                m.save(netPath / ('net%i' % i))
            index_name = self.parameters.index.name
            self.parameters.loc["nMaxPool"] = self.nMaxPool
            if any(self.parameters.index=="maxL"):
                self.parameters.loc["maxL"] = self.maxL
            else:
                self.parameters = self.parameters.append(pd.DataFrame({"Parameter_Value":self.maxL},index=("maxL",)))
            self.parameters.index.name = index_name
            self.parameters.to_csv(modelPath / 'parameters.txt',index_label = self.parameters.index.name)
            data2save = self.trainingData
            if hasattr(self,'trainingDataModel'):
                data2save = data2save + self.trainingDataModel
            pd.DataFrame(data2save).to_csv(modelPath / "".join(('data_',self.allele,'.txt')),index=None,header=None)
        return models
    
    def applyCNN(self,models,IM,saveOutcome=False):
        nbRep =len(models)
        N = IM.shape[0]
        yhat = np.zeros((N,1)) 
        for i in np.arange(0,nbRep):
            yhat += models[i].predict(IM)
        yhat = yhat[:,0]/nbRep
        if saveOutcome:
            self.predictedOutcomeApply = yhat
        return yhat
                
    def crossValCNN(self,IM,out):
        time_start = time.perf_counter()
        C = self.cvPart
        modelCV = []
        N = IM.shape[0]
        yhatCV = np.repeat(np.nan,N)
        if hasattr(self,'trainingDataModel'):
            trainingDataModel_lmer = self.int2aa(self.seq2Lmer(self.aa2int(self.trainingDataModel),L=self.l)[0])
            ind2rmv = np.array([any(s2 in s1 for s2 in trainingDataModel_lmer) for s1 in self.trainingData]).nonzero()[0]
        else:
            ind2rmv = []
        for i,c in enumerate(np.unique(C)):
            print('Doing: fold %i \n' % i)
            indTest = np.where(C==c)[0]
            indTest = np.setdiff1d(indTest,ind2rmv)
            indTrain = np.where(C!=c)[0]
            modelCVcurr = self.trainCNN(IM[indTrain,:,:,:], out[indTrain])
            yhatCV[indTest] = self.applyCNN(modelCVcurr,IM[indTest,:,:,:])
            modelCV.append(modelCVcurr)
            print('fold %i: Done \n' % i)
        time_elapsed = (time.perf_counter() - time_start)
        self.predictedOutcomeCV = yhatCV
        self.modelCV = modelCV
        self.timeCV = time_elapsed
        return yhatCV,modelCV
    
    def feedForwardAndGetScore(self,seq,saveOutcome=False):
        time_start = time.perf_counter()
        nbSamples = len(seq)
        maxNbSamples = self.maxNbSamples2apply
        nbNets = len(self.trainedModels)
        l = self.l
        h,w = self.trainedModels[0].input_shape[1:3]
        nMaxPool = self.trainedModels[0].layers[1].pool_size[0]
        sizeOutConv2D = h-l+1
        sizeOutMaxPool = sizeOutConv2D - nMaxPool + 1
        contributionScore = np.array([]).reshape((sizeOutConv2D,0))
        yhat = []
        for j in np.arange(0,int(np.ceil(nbSamples/maxNbSamples))):
            nSamples = np.min((maxNbSamples,len(seq)))
            seq1 = seq[0:nSamples]
            seq = seq[nSamples:None]
            IM = self.getImages(seq1)
            IM = tf.constant(IM)
            yhat1 = np.zeros((nSamples,nbNets))
            contributionScore1 = np.zeros((sizeOutConv2D,nSamples,nbNets))
            for i in np.arange(0,nbNets):
                W1 = tf.constant(self.trainedModels[i].layers[0].get_weights()[0])
                b1 = tf.constant(self.trainedModels[i].layers[0].get_weights()[1])
                W2 = tf.constant(self.trainedModels[i].layers[3].get_weights()[0])
                b2 = tf.constant(self.trainedModels[i].layers[3].get_weights()[1])
                actFct = self.trainedModels[i].layers[3].activation
                F = W1.shape[3]
                outConv2D=tf.nn.conv2d(IM, W1, strides=[1, 1, w, 1], padding='VALID')+b1
                outReLu = tf.nn.relu(outConv2D)
                resultMaxPool=tf.nn.max_pool_with_argmax(outReLu, [1,nMaxPool,1,1], [1,1,1,1], padding='VALID')
                out_sum=tf.multiply(tf.reshape(resultMaxPool[0],(-1,W2.shape[0])),tf.transpose(tf.tile(W2,(1,nSamples))))
                argMaxPool2D = (resultMaxPool[1]-np.tile(np.reshape(np.arange(0,F),(1,1,1,F)),(nSamples,sizeOutMaxPool,1,1)))/F
                argMaxPool2D = tf.cast(argMaxPool2D,'int32')
                toAdd=np.tile(np.reshape(np.arange(0,nSamples)*sizeOutConv2D,(nSamples,1,1,1)),(1,sizeOutMaxPool,1,F))
                argMaxPool2D = argMaxPool2D+toAdd
                contributionScoreCurr = tf.math.unsorted_segment_sum(tf.reshape(out_sum,(-1,)),tf.reshape(argMaxPool2D,(-1,)),sizeOutConv2D*nSamples)
                contributionScoreCurr = tf.reshape(contributionScoreCurr,(nSamples,sizeOutConv2D))
                yhatCurr = actFct(tf.reduce_sum(contributionScoreCurr,1)+b2)
                yhat1[:,i] = np.reshape(yhatCurr.numpy(),(nSamples,))
                contributionScore1[:,:,i] = contributionScoreCurr.numpy().T    
            yhat1 = np.mean(yhat1,1)
            contributionScore1 = np.sum(contributionScore1,axis=2)
            contributionScore1 = contributionScore1/np.tile(np.sum(contributionScore1,axis=0),(sizeOutConv2D,1))
            contributionScore = np.concatenate((contributionScore,contributionScore1),1)
            yhat = np.concatenate((yhat,yhat1))
        time_elapsed = (time.perf_counter() - time_start)
        if saveOutcome:
            self.contributionScore = contributionScore
            self.predictedOutcomeWithContributionScore = yhat
            self.timeApply = time_elapsed
        return contributionScore,yhat
        
    def generateRandomSeq(self,followLengthDistr=False):
        nbRandSeq = self.nbRandSeq
        nbPrev = self.nbPrev
        nbAfter = self.nbAfter
        maxL = self.maxL
        if followLengthDistr:
            seqL = self.seqL
            N = np.histogram(seqL,range(1,maxL+2))[0]
            a = nbRandSeq*N/np.sum(N)
            nbRandSeqPerLength = np.array([np.int32(np.round(a[i])) for i in range(0,len(a))])
            zeroL = np.where(nbRandSeqPerLength==0.)
            L = np.arange(0,maxL)
            nbRandSeqPerLength = np.delete(nbRandSeqPerLength,zeroL)
            L = np.delete(L,zeroL)
            nbRandSeq = np.sum(nbRandSeqPerLength)
            sR = np.random.randint(0,high=19,size=(nbRandSeq,maxL+nbAfter+nbPrev))
            k = 0
            for i in np.arange(0,len(nbRandSeqPerLength)):
                currL = L[i]
                currN = nbRandSeqPerLength[i]
                sR[k:(k+currN),(currL+1+nbPrev):(maxL+nbAfter+nbPrev)] = 20
                sR[k:(k+currN),0:nbPrev] = 20
                k = k+currN
        else:
            randSeqLength = self.lengthRandSeq
            sR = np.concatenate((20*np.ones((nbRandSeq,nbPrev),dtype=int),np.random.randint(0,high=19,size=(nbRandSeq,randSeqLength)),20*np.ones((nbRandSeq,maxL+nbAfter-randSeqLength),dtype=int)),axis=1)
        self.randSeq = sR
        return sR
    
    def plotLogoSeq(self,contributionScore,yhat):
        l = self.l
        sR = self.randSeq
        nbBest = self.nbBest
        indBest = np.flip(np.argsort(yhat))[0:nbBest]
        contributionScore = contributionScore[:,indBest]
        mm = np.argmax(contributionScore,0)
        sB = np.zeros((nbBest,l))
        sB = []
        for i,ind in enumerate(indBest):
            sB.append(sR[ind][mm[i]:mm[i]+l])
        sBchar = self.int2aa(sB)
        pfm = lm.alignment_to_matrix(sBchar)
        pim = lm.transform_matrix(pfm,from_type='counts',to_type='information')
        h = lm.Logo(pim,color_scheme='hydrophobicity')
        h.ax.set_title(self.allele)
        h.ax.xaxis.set_ticks(np.array(np.arange(0,l),dtype=float))
        h.ax.xaxis.set_ticklabels(np.arange(1,l+1))
        name = [self.allele,'_logoPlot.png']
        h.fig.savefig(self.savePath / "".join(name))
        self.seqInLogoPlot = sBchar
        self.pimLogoPlot = pim
        return h,sBchar,pim
    
    def computationTime(self,time_elapsed):
        self.timeTotal = time_elapsed
        
    def getCVresults(self):
        yhatCV = self.predictedOutcomeCV
        ind_no_nan = ~np.isnan(yhatCV)
        out = self.trainingOutcome[ind_no_nan]
        yhatCV = yhatCV[ind_no_nan]
        thr = self.bindingThr
        nbRound = 3
        outBin = out>thr
        yhatBin = yhatCV>thr
        PC = np.corrcoef(yhatCV,out)[0,1].round(nbRound)
        mse = sklearn.metrics.mean_squared_error(out, yhatCV)
        rmse = np.array(math.sqrt(mse)).round(nbRound)
        if len(np.unique(outBin))>1:
            AUC = np.array(sklearn.metrics.roc_auc_score(outBin,yhatCV)).round(nbRound)
            mcc = np.array(sklearn.metrics.matthews_corrcoef(outBin,yhatBin)).round(nbRound)
            acc = np.array(sklearn.metrics.accuracy_score(outBin,yhatBin)).round(nbRound)
            bacc = np.array(sklearn.metrics.balanced_accuracy_score(outBin,yhatBin)).round(nbRound)
            f1 = np.array(sklearn.metrics.f1_score(outBin,yhatBin)).round(nbRound)
        else:
            AUC = None
            mcc = None
            acc = None
            bacc = None
            f1 = None
        resultsCV = pd.DataFrame([self.allele,len(out),np.sum(outBin),PC,AUC,rmse,mcc,acc,bacc,f1]).transpose()
        resultsCV.columns = ['Allele','#Peptide','#Binder','PC','AUC','RMSE','MCC','ACC','BACC','F1']
        resultsCV.to_csv(self.savePath / 'cross_validation_results.txt',index=False)
        self.resultsCV = resultsCV
        
    def printApplyOutcome(self,saveTable = False):
        sApply = np.array(self.int2aa(self.applyDataSeq))
        nameApply = self.applyDataSeqName
        yhatApply = self.predictedOutcomeWithContributionScore
        yhatApply = yhatApply.round(3)
        coreBinders = self.coreBinders
        posStart = self.applyDataPositionStart 
        posEnd = self.applyDataPositionEnd
        indS = np.flip(np.argsort(yhatApply))
        if nameApply is not None:
            table = pd.DataFrame([nameApply[indS],posStart[indS],posEnd[indS],sApply[indS],coreBinders[indS],yhatApply[indS]]).transpose()
            table.columns = ['Peptide_Source','Start','End','Peptide','Binding_Core','Predicted_Outcome']
        else:
            table = pd.DataFrame([posStart[indS],posEnd[indS],sApply[indS],coreBinders[indS],yhatApply[indS]]).transpose()
            table.columns = ['Start','End','Peptide','Binding_Core','Predicted_Outcome']
        table = table.drop_duplicates('Binding_Core')
        name = [self.allele,'_predictedOutcome.txt']
        table.to_csv(self.savePath / "".join(name),index=False)
        if saveTable:
            self.tablePrediction = table
        return table
    
    def seq2Lmer(self,seq,L=None,nameSeq=None,takeUniqueLmer=True,saveLmer=False):
        if isinstance(nameSeq,list):
            isNameSeqNone = False
            nameSeq = np.array(nameSeq)
        else:
            isNameSeqNone = np.any(nameSeq==None)
        if L==None:
            L = self.epitopesLength
        I = lambda x: np.tile(np.arange(0,L),x)+np.repeat(np.arange(0,x),L)
        lmer_length = [max((0,len(s)-L+1)) for s in seq]
        N = sum(lmer_length)
        sLmer = np.empty((N,L),dtype=int)
        indLmer = np.empty((N,),dtype=int)
        pos1 = indLmer.copy()
        k=0
        for i,s in enumerate(seq):
            if lmer_length[i]:
                ind2fill = np.arange(k,k+lmer_length[i])
                sLmer[ind2fill,:] = s[I(lmer_length[i])].reshape((-1,L))
                k += lmer_length[i]
                indLmer[ind2fill] = np.tile(i,lmer_length[i])
                pos1[ind2fill] = np.arange(1,lmer_length[i]+1)
        if takeUniqueLmer:
            sLmer,indU = np.unique(sLmer,return_index=True,axis=0)
            indLmer = indLmer[indU]
            pos1 = pos1[indU]
        if isNameSeqNone:
            nameSeqLmer = None
        else:
            nameSeqLmer = nameSeq[indLmer]
        pos2 = pos1 + L - 1
        if saveLmer:
            self.applyDataSeq = sLmer
            self.applyDataSeqName = nameSeqLmer
            self.applyDataPositionStart = pos1
            self.applyDataPositionEnd = pos2
        return sLmer,nameSeqLmer,indLmer

    def getCoreBinder(self,seq,contributionScore,applyDataName=None,saveCoreBinders=False):
        l = self.l
        indMax = np.argmax(contributionScore,0)
        sCore = np.array([seq[i][indMax[i]:indMax[i]+l] for i in np.arange(len(seq))])
        if saveCoreBinders:
            self.coreBinders = sCore
        return sCore
    
    def save_object(self,name=None):
        if name is None:
            name = (self.allele,'_ModelCNN.pkl')
            name = "".join(name)
        sPath = self.savePath / ('model_'+self.allele)
        if not os.path.isdir(sPath):
            os.mkdir(sPath)
        filename = sPath / name
        if hasattr(self,'trainedModels'):
            delattr(self,'trainedModels')
        delattr(self,'savePath')
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        self.savePath = sPath
            
    def load_object(self,filename):
        with open(filename,'rb') as output:
            obj = pickle.load(output)
        filename = Path(filename)
        obj.savePath = filename.parent
        return obj
    
    def feedForwardVisualization(self,s,fontSize=4,dpi=300):
        import matplotlib.pyplot as plt
        if isinstance(s,str):
            s = [s]
        seq = self.aa2int(s)
        seq = self.addEmptyPositions(seq)
        seqAA = self.int2aa(seq)
        model = self.trainedModels
        nbPrev = self.nbPrev
        nbAfter = self.nbAfter
        maxL = self.maxL
        nbSamples = len(seq)
        maxNbSamples = self.maxNbSamples2apply
        nbNets = len(model)
        l = self.l
        folderName = self.savePath / 'feed_forward_visualization'
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        folderNameSeq = folderName / 'sequences'
        if not os.path.exists(folderNameSeq):
            os.makedirs(folderNameSeq)
        folderNameNets = folderName / 'nets'
        if not os.path.exists(folderNameNets):
            os.makedirs(folderNameNets)
        h,w = model[0].input_shape[1:3]
        nMaxPool = model[0].layers[1].pool_size[0]
        sizeOutConv2D = h-l+1
        sizeOutMaxPool = sizeOutConv2D - nMaxPool + 1
        yhat = []
        actFct = getattr(tf.keras.activations,self.activationFctDenseLayer)
        for j in range(0,int(np.ceil(nbSamples/maxNbSamples))):
            nSamples = np.min((maxNbSamples,len(seq)))
            seq1 = seq[0:nSamples]
            s1 = s[0:nSamples]
            nSamples = np.min((maxNbSamples,len(seq)))
            seq1 = seq[0:nSamples]
            s1 = s[0:nSamples]
            folderNamesAllSeq = []
            seq = seq[nSamples:None]
            s = s[nSamples:None]
            IM = self.getImages(seq1)
            seq_length = []
            for s_i,currSeq in enumerate(s1):
                seq_length.append(len(currSeq))
                IMcurr = IM[s_i,:,:,0]
                seqAAcurr = [char for char in seqAA[s_i]]
                folderNameSeqCurr = folderNameSeq / currSeq
                folderNamesAllSeq.append(folderNameSeqCurr)
                if not os.path.exists(folderNameSeqCurr):
                    os.makedirs(folderNameSeqCurr)
                fileNameCurr = folderNameSeqCurr / 'input_image.png'
                if not os.path.exists(fileNameCurr):
                        h=plt.imshow(IMcurr,cmap='gray')
                        h.axes.xaxis.set_ticks(np.array(np.arange(0,21),dtype=float))
                        h.axes.xaxis.set_ticklabels(['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','-'])
                        h.axes.yaxis.set_ticks(np.array(np.arange(0,IMcurr.shape[0]),dtype=float))
                        h.axes.yaxis.set_ticklabels(seqAAcurr)
                        for tick in h.axes.xaxis.get_major_ticks():
                            tick.label.set_fontsize(fontSize)
                        for tick in h.axes.yaxis.get_major_ticks():
                            tick.label.set_fontsize(fontSize)
                        h.figure.savefig(fileNameCurr,dpi = (dpi))
                        h.figure.clear()
                        plt.close(h.figure)
            IM = tf.constant(IM)
            yhat1 = np.zeros((nSamples,nbNets))
            for i in range(0,nbNets):
                i1 = i+1
                folderNameNetCurr = folderNameNets / ('net_%i' % i1)
                if not os.path.exists(folderNameNetCurr):
                    os.makedirs(folderNameNetCurr) 
                folderNameNetConv2DCurr = folderNameNetCurr / 'Conv2D_layer'
                if not os.path.exists(folderNameNetConv2DCurr):
                    os.makedirs(folderNameNetConv2DCurr)
                folderNameNetDenseCurr = folderNameNetCurr / 'Dense'
                if not os.path.exists(folderNameNetDenseCurr):
                    os.makedirs(folderNameNetDenseCurr)
                W1 = tf.constant(model[i].layers[0].get_weights()[0])
                b1 = tf.constant(model[i].layers[0].get_weights()[1])
                F = W1.shape[3]
                W1np = W1.numpy()
                vmin = np.min(W1np)
                vmax = np.max(W1np)
                F_name = []
                for f in range(0,F):
                    f1 = f+1
                    F_name.append(('F%i' % f1))
                    fileNameCurr = folderNameNetConv2DCurr / ('filter_%i.png' % f1)
                    if not os.path.exists(fileNameCurr):
                        Wcurr = W1np[:,:,0,f]
                        h=plt.imshow(Wcurr,cmap='gray',vmin=vmin,vmax=vmax)
                        h.axes.xaxis.set_ticks(np.array(np.arange(0,21),dtype=float))
                        h.axes.xaxis.set_ticklabels(['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','-'])
                        h.axes.yaxis.set_ticks(np.array(np.arange(0,l),dtype=float))
                        h.axes.yaxis.set_ticklabels(np.array(np.arange(1,l+1),dtype=int))
                        for tick in h.axes.xaxis.get_major_ticks():
                            tick.label.set_fontsize(fontSize)
                            tick.label.set_rotation('horizontal')
                        for tick in h.axes.yaxis.get_major_ticks():
                            tick.label.set_fontsize(fontSize)
                        h.figure.savefig(fileNameCurr,dpi=(dpi))
                        h.figure.clear()
                        plt.close(h.figure)
                W2 = tf.constant(model[i].layers[3].get_weights()[0])
                b2 = tf.constant(model[i].layers[3].get_weights()[1])
                W2np = W2.numpy().reshape((-1,F))
                vmin = np.min(W2np)
                vmax = np.max(W2np)
                fileNameCurr = folderNameNetDenseCurr / 'weights_dense.png'
                if not os.path.exists(fileNameCurr):
                        h=plt.imshow(W2np,cmap='gray',vmin=vmin,vmax=vmax)
                        h.axes.xaxis.set_ticks(np.array(np.arange(0,F),dtype=float))
                        h.axes.xaxis.set_ticklabels(F_name)
                        h.axes.yaxis.set_ticks([])
                        h.axes.yaxis.set_ticklabels([])
                        for tick in h.axes.xaxis.get_major_ticks():
                            tick.label.set_fontsize(fontSize)
                            tick.label.set_rotation('vertical')
                        for tick in h.axes.yaxis.get_major_ticks():
                            tick.label.set_fontsize(fontSize)
                        h.figure.savefig(fileNameCurr,dpi=(dpi))
                        h.figure.clear()
                        plt.close(h.figure)
                outConv2D=tf.nn.conv2d(IM, W1, strides=[1, 1, w, 1], padding='VALID')+b1
                outReLu = tf.nn.relu(outConv2D)
                outReLu_np = outReLu.numpy()[:,:,0,:]
                resultMaxPool=tf.nn.max_pool_with_argmax(outReLu, [1,nMaxPool,1,1], [1,1,1,1], padding='VALID')
                outMaxPool_np = resultMaxPool[0].numpy()[:,:,0,:]
                argMaxPool2D = (resultMaxPool[1]-np.tile(np.reshape(np.arange(0,F),(1,1,1,F)),(nSamples,sizeOutMaxPool,1,1)))/F
                argMaxPool2D = tf.cast(argMaxPool2D,'int32')
                argMaxPool2D_np = argMaxPool2D.numpy()[:,:,0,:]
                
                yhatCurr = actFct(tf.math.reduce_sum(tf.math.multiply(tf.transpose(tf.reshape(resultMaxPool[0],(nSamples,-1))),tf.tile(W2,(1,nSamples))),axis=0)+b2)
                yhat1[:,i] = np.reshape(yhatCurr.numpy(),(nSamples,))
                for s_i,seqFold in enumerate(folderNamesAllSeq):
                    folderNameSeqNetCurr = seqFold / ('net_%i' % i1)
                    if not os.path.exists(folderNameSeqNetCurr):
                        os.makedirs(folderNameSeqNetCurr)
                    nbLmers = maxL+nbPrev+nbAfter-l+1
                    nbSeqLmers = seq_length[s_i]-l+1
                    lmersLabel = ['p_%i' % ind_lmer for ind_lmer in np.flip(np.arange(1,nbPrev+1))]
                    for ind_lmer in np.arange(1,nbSeqLmers+1):
                        lmersLabel.append('l_%i' % ind_lmer)
                    for ind_lmer in np.arange(1,nbLmers-nbSeqLmers-nbPrev+1):
                        lmersLabel.append('a_%i' % ind_lmer)
                    vmin = np.min(outReLu_np[s_i])
                    vmax = np.max(outReLu_np[s_i])
                    fileNameCurr = folderNameSeqNetCurr / 'conv2D_output.png'
                    if not os.path.exists(fileNameCurr):
                            h=plt.imshow(outReLu_np[s_i],cmap='gray',vmin=vmin,vmax=vmax)
                            h.axes.xaxis.set_ticks(np.array(np.arange(0,F),dtype=float))
                            h.axes.xaxis.set_ticklabels(F_name)
                            h.axes.yaxis.set_ticks(np.array(np.arange(0,nbLmers),dtype=float))
                            h.axes.yaxis.set_ticklabels(lmersLabel)
                            for tick in h.axes.xaxis.get_major_ticks():
                                tick.label.set_fontsize(fontSize)
                                tick.label.set_rotation('vertical')
                            for tick in h.axes.yaxis.get_major_ticks():
                                tick.label.set_fontsize(fontSize)
                            h.figure.savefig(fileNameCurr,dpi=(dpi))
                            h.figure.clear()
                            plt.close(h.figure)
                    argMaxPool2Dcurr = argMaxPool2D_np[s_i,:,:]
                    lmersTable = []
                    for cell in argMaxPool2Dcurr.reshape((-1,1)):
                       lmersTable.append(lmersLabel[int(cell)])  
                    lmersTable = np.array(lmersTable).reshape(argMaxPool2Dcurr.shape)
                    lmersTable = pd.DataFrame(lmersTable)
                    lmersTable.columns = F_name
                    vmin = np.min(outMaxPool_np[s_i])
                    vmax = np.max(outMaxPool_np[s_i])
                    fileNameCurr = folderNameSeqNetCurr / 'maxPool_output.png'
                    if not os.path.exists(fileNameCurr):
                            h=plt.imshow(outMaxPool_np[s_i],cmap='gray',vmin=vmin,vmax=vmax)
                            h.axes.xaxis.set_ticks(np.array(np.arange(0,F),dtype=float))
                            h.axes.xaxis.set_ticklabels(F_name)
                            h.axes.yaxis.set_ticks([])
                            h.axes.yaxis.set_ticklabels([])
                            for tick in h.axes.xaxis.get_major_ticks():
                                tick.label.set_fontsize(fontSize)
                                tick.label.set_rotation('vertical')
                            for tick in h.axes.yaxis.get_major_ticks():
                                tick.label.set_fontsize(fontSize)
                            h.figure.savefig(fileNameCurr,dpi=(dpi))
                            h.figure.clear()
                            plt.close(h.figure)
                            lmersTable.to_html(folderNameSeqNetCurr / 'maxPool_output_arguments.html',index=False)
                
            yhat1 = np.mean(yhat1,1)
            yhat = np.concatenate((yhat,yhat1))
        return yhat
    
    def generateCVpartWithLeastLmerOverlap(self,kFold,saveCVPart=False):
        def countSharedlmers(C,sLmer,indLmer,indPosLmer):
            CLmer = [C[i][0] for i in indLmer]
            Cu = np.unique(C)
            k = Cu.shape[0]
            lmersPerPartPos = []
            lmersPerPartNeg = []
            for c in Cu:
                ind2take = np.where(CLmer==c)[0]
                ind2takePos = np.intersect1d(ind2take,np.where(indPosLmer)[0])
                ind2takeNeg = np.intersect1d(ind2take,np.where(indPosLmer==False)[0])
                lmersPerPartPos.append([sLmer[ci] for ci in ind2takePos])
                lmersPerPartNeg.append([sLmer[ci] for ci in ind2takeNeg])
            lmersInterPos = []
            lmersInterNeg = []
            count = 0
            for i in range(0,k):
                 lmersPerPartPosTest = lmersPerPartPos[i]
                 lmersPerPartNegTest = lmersPerPartNeg[i]
                 trainFolds = np.delete(np.arange(0,k),i).tolist()
                 lmersPerPartPosTrain = [lmersPerPartPos[ic] for ic in trainFolds]
                 lmersPerPartPosTrain = [jj for ii in lmersPerPartPosTrain for jj in ii]
                 lmersPerPartNegTrain = [lmersPerPartNeg[ic] for ic in trainFolds]
                 lmersPerPartNegTrain = [jj for ii in lmersPerPartNegTrain for jj in ii]
                 lPos = list(set(lmersPerPartPosTrain).intersection(lmersPerPartPosTest))
                 lNeg = list(set(lmersPerPartNegTrain).intersection(lmersPerPartNegTest))
                 [lmersInterPos.append(el) for el in lPos]
                 [lmersInterNeg.append(el) for el in lNeg]
                 count+=len(lPos)+len(lNeg)
            lmersInter = list(set(lmersInterPos+lmersInterNeg))
            return count/k,lmersInter
        s = self.trainingData
        indPos = self.trainingOutcome>self.bindingThr
        sLmer,a,indLmer = self.seq2Lmer(self.aa2int(s),L=self.l,takeUniqueLmer=False,saveLmer=False)
        sLmer = self.int2aa(sLmer)
        indPosLmer = [indPos[i] for i in indLmer]
        nbSeq = len(s)
        
        nbPerFold = int(np.floor(nbSeq/kFold))
        nbRest = int(nbSeq-kFold*nbPerFold)
        cvPart = np.array([],dtype=int).reshape(0,1)
        for i in range(0,kFold):
            cvPart = np.concatenate((cvPart,i*np.ones((nbPerFold,1),dtype=int)))
        cvPart = np.concatenate((cvPart,np.random.choice(range(kFold),size=(nbRest,1),replace=False)))
        cvPart = np.random.permutation(cvPart)
        
        lmerInter = countSharedlmers(cvPart,sLmer,indLmer,indPosLmer)[1]
        if len(lmerInter)>0:
            s = self.int2aa(self.aa2int(s))
        for ss in lmerInter:
            indWithLmer = [sc.find(ss)>-1 for sc in s]
            currCount = np.bincount(cvPart[indWithLmer].reshape((-1,)))
            cvInd = np.random.permutation(np.where(currCount==np.max(currCount))[0])[0]
            cvPart[indWithLmer] = cvInd

        averageLmersOverlappingCV = countSharedlmers(cvPart,sLmer,indLmer,indPosLmer)[0]
        if saveCVPart:
            self.cvPart = cvPart
            self.averageLmersOverlappingCV = averageLmersOverlappingCV
        return cvPart,averageLmersOverlappingCV