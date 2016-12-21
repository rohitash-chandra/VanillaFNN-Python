# Rohitash Chandra, 2017 c.rohitash@gmail.conm

#!/usr/bin/python

# ref: http://iamtrask.github.io/2015/07/12/basic-python-network/  

#con grad: http://andrew.gibiansky.com/blog/machine-learning/conjugate-gradient/


#Sigmoid units used in hidden and output layer. gradient descent and stocastic gradient descent functions implemented with momentum. Note:
#  Classical momentum:

#vW(t+1) = momentum.*Vw(t) - scaling .* gradient_F( W(t) )
#W(t+1) = W(t) + vW(t+1)

#W Nesterov momentum is this: http://cs231n.github.io/neural-networks-3/

#vW(t+1) = momentum.*Vw(t) - scaling .* gradient_F( W(t) + momentum.*vW(t) )
#W(t+1) = W(t) + vW(t+1)

#http://jmlr.org/proceedings/papers/v28/sutskever13.pdf

#no bias is implenmented 

# Numpy used: http://cs231n.github.io/python-numpy-tutorial/#numpy-arrays

#Rprop: http://peterroelants.github.io/posts/rnn_implementation_part01/


from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model 

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import   wilcoxon, ttest_ind, mannwhitneyu 


import random
import time

#An example of a class
class Network:

    def __init__(self, Topo, Train, Test, MaxTime, Samples, MinPer):
     

        self.epsilon = 0.1 # learning rate for gradient descent
        self.reg_lambda = 0.1 # regularization strength
        self.Top  = Topo  # NN topology [input, hidden, output]
        self.Max = MaxTime # max epocs
        self.TrainData = Train
        self.TestData = Test
        self.NumSamples = Samples

        self.lrate  = 0 # will be updated later with BP call

        self.momenRate = 0
        self.useNesterovMomen = 0 #use nestmomentum 1, not use is 0

        self.minPerf = MinPer
                                        #initialize weights ( W1 W2 ) and bias ( b1 b2 ) of the network
    	np.random.seed(0) 
   	self.W1 = np.random.randn(self.Top[0], self.Top[1])  / np.sqrt(self.Top[0])
      #  print self.W1
        self.BestW1 = self.W1
        #self.b1 = np.zeros((1, self.Top[1]))
    	self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.BestW2 = self.W2
  	#self.b2 = np.zeros((1, self.Top[2]))
        self.hidout = np.zeros((1, self.Top[1])) # output of first hidden layer
        self.out = np.zeros((1, self.Top[2])) #  output last layer

  
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self,actualout):
        error = np.subtract(self.out, actualout)
        sqerror= np.sum(np.square(error))/self.Top[2] 
        #print sqerror
        return sqerror
  
    def ForwardPass(self, X ):
         z1 = X.dot(self.W1) #+ self.b1
         self.hidout = self.sigmoid(z1) # output of first hidden layer
         z2 = self.hidout.dot(self.W2)# + self.b2
         self.out = self.sigmoid(z2)  # output second hidden layer
  	   

    def BackwardPassVanilla(self, Input, desired):  #implements   Vanilla BP (Canonical BP)
       	  out_delta =   (desired - self.out)*(self.out*(1-self.out)) 
          hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1-self.hidout)) 
          self.W2+= (self.hidout.T.dot(out_delta) * self.lrate)  
          self.W1 += (Input.T.dot(hid_delta) * self.lrate) 
  
    def BackwardPassMomentum(self, Input, desired):   
            out_delta =   (desired - self.out)*(self.out*(1-self.out)) 
            hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1-self.hidout))
        
 	    v2 = self.W2 #save previous weights http://cs231n.github.io/neural-networks-3/#sgd
	    v1 = self.W1 

            v2 = ( v2 *self.momenRate) + (self.hidout.T.dot(out_delta) * self.lrate)       # velocity update
            v1 = ( v1 *self.momenRate) + (Input.T.dot(hid_delta) * self.lrate)  

            if self.useNesterovMomen == 0: # use classical momentum 
               self.W2+= v2
       	       self.W1 += v1 

            else: # useNesterovMomen http://cs231n.github.io/neural-networks-3/#sgd
               v2_prev = v2
               v1_prev = v1  
	       self.W2+= (self.momenRate * v2_prev + (1 + self.momenRate) )  * v2
       	       self.W1 += ( self.momenRate * v1_prev + (1 + self.momenRate) )  * v1 

          

        

 
          
    def compare_out(Out, Desired, erToler):
      #traverse and check
        return 0




    def TestNetwork(self, Data, testSize, erTolerance):
        Input = np.zeros((1, self.Top[0])) # temp hold input
        Desired = np.zeros((1, self.Top[2])) 
        nOutput = np.zeros((1, self.Top[2]))
        clasPerf = 0
     	sse = 0  
        self.W1 = self.BestW1
        self.W2 = self.BestW2 #load best knowledge
     
        for s in xrange(0, testSize):
                  
                Input[:]  =   Data[s,0:self.Top[0]] 
                Desired[:] =  Data[s,self.Top[0]:] 

                self.ForwardPass(Input) 
                sse = sse+ self.sampleEr(Desired) 
               # print sse

              #  print self.out 
               # print Desired
               # print(s) 

                #if(compareMethod(self.out, Desired, erTolerance)):


                if(np.isclose(self.out, Desired, atol=erTolerance).any()):
                   clasPerf =  clasPerf +1 
                #   print clasPerf

   	return ( sse/testSize, float(clasPerf)/testSize * 100 )

 
    def saveKnowledge(self):
        self.BestW1 = self.W1
        self.BestW2 = self.W2 
 
    def BP_GD(self, learnRate, mRate,  useNestmomen , stocastic): # BP with SGD (Stocastic BP)
        self.lrate = learnRate
        self.momenRate = mRate
        self.useNesterovMomen =  useNestmomen  
     
        Input = np.zeros((1, self.Top[0])) # temp hold input
        Desired = np.zeros((1, self.Top[2])) 
        Er = []#np.zeros((1, self.Max)) 
        epoch = 0
        bestmse = 100
        bestTrain = 0
        while  epoch < self.Max and bestTrain < self.minPerf :
            #print(epoch)
            sse = 0
            for s in xrange(0, self.NumSamples):
              
                if(stocastic):
                   pat = random.randint(0, self.NumSamples-1) 
                else:
                   pat = s 

                Input[:]  =  self.TrainData[pat,0:self.Top[0]] 
                Desired[:] = self.TrainData[pat,self.Top[0]:]

                self.ForwardPass(Input) 
        	#self.BackwardPassVanilla(Input, Desired)
                self.BackwardPassMomentum(Input, Desired)
                sse = sse+ self.sampleEr(Desired)
             
            mse = np.sqrt(sse/self.NumSamples)

            if mse < bestmse:
               bestmse = mse
               self.saveKnowledge() 
               (x,bestTrain) = self.TestNetwork(self.TrainData, self.NumSamples, 0.2)
              # print(bestmse,x,bestTrain)

            Er = np.append(Er, mse)

            #print(mse, bestmse)

            epoch=epoch+1 
            #print(self.W1)
            #print(self.BestW1)

        return (Er,bestmse, bestTrain, epoch) 
 

def main(): 

        problem = 1

        if problem == 1:
 	   TrainData = np.loadtxt("train.csv", delimiter=',') #  
           TestData = np.loadtxt("test.csv", delimiter=',') #  
  	   Hidden = 8
           Input = 4
           Output = 2
           TrSamples =  110
           TestSize = 40
           learnRate = 0.1 
           mRate = 0.01
           useNestmomen = 0

        if problem == 2:
 	   TrainData = np.loadtxt("4bit.csv", delimiter=',') #  
           TestData = np.loadtxt("4bit.csv", delimiter=',') #  
  	   Hidden = 6
           Input = 4
           Output = 1
           TrSamples =  16
           TestSize = 16
           learnRate = 0.1 
           mRate = 0.01
           useNestmomen = 1


 
        #print(TrainData)
        
        #print(TestData)
  
 
    

        Topo = [Input, Hidden, Output] 
        MaxRun = 10 # number of experimental runs 
        MaxTime = 200 
        MinCriteria = 95 #stop when learn 95 percent
        
        trainTolerance = 0.4
        testTolerance = 0.2
        
        useStocasticGD = 1 # 0 for vanilla BP. 1 for Stocastic BP
       
        trainPerf = np.zeros(MaxRun)
        testPerf =  np.zeros(MaxRun)

        trainMSE =  np.zeros(MaxRun)
        testMSE =  np.zeros(MaxRun)
        Epochs =  np.zeros(MaxRun)
        Time =  np.zeros(MaxRun)

        for run in xrange(0, MaxRun  ): 
                 print run
                 fnnSGD = Network(Topo, TrainData, TestData, MaxTime, TrSamples, MinCriteria) # Stocastic GD
        	 start_time=time.time()
                 (erEp,  trainMSE[run] , trainPerf[run] , Epochs[run]) = fnnSGD.BP_GD(learnRate, mRate, useNestmomen,  useStocasticGD)   

                 Time[run]  =time.time()-start_time
                 (testMSE[run], testPerf[run]) = fnnSGD.TestNetwork(TestData, TestSize, testTolerance)
                
        print trainPerf 
        print testPerf
        print trainMSE
        print testMSE

        print Epochs
        print Time
        print(np.mean(trainPerf), np.std(trainPerf))
        print(np.mean(testPerf), np.std(testPerf))
        print(np.mean(Time), np.std(Time))
 	
  
         
 	plt.figure()
	plt.plot(erEp )
	plt.ylabel('error')  
        plt.savefig('out.png')
       
 
if __name__ == "__main__": main()


