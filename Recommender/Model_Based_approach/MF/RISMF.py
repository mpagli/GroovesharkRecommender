#!/usr/bin/python

import sys 
import random as rnd 
import numpy as np 

class RISMF(object):
	"""Implementation of Regularized Incremental Simultaneous MF. Simple version of gradient descent based MF with weight decay"""

	def __init__(self,trainSet,numItems,numUsers,weightDecay,learningRate,numFeatures):
		"""the training and validation set are formatted as follow: [(u0,i0,r0),(u4,i4,r4), ...] as a list of (userId,itemId,rating) tuples"""
		self.trainSet = trainSet
		self.validSet = None
		self.weightDecay = weightDecay
		self.learningRate = learningRate
		self.numFeatures = numFeatures
		self.numItems = numItems
		self.numUsers = numUsers
		self.P  = np.abs(np.random.randn(self.numUsers,self.numFeatures).astype(np.float64))
		self.Q  = np.abs(np.random.randn(self.numFeatures,self.numItems).astype(np.float64))
		self.PQ = None
		self.sampleTrainSet()

	def save(self,dirPath):
		np.save(dirPath+"Q.npy",self.Q)

	def sampleTrainSet(self,fraction = 0.2):
		"""generate the validation set by picking element in the training set at random"""
		print "trainSet size:",len(self.trainSet)
		self.validSet = []
		numToRemove = int(fraction*len(self.trainSet))
		indices = range(len(self.trainSet))
		rnd.shuffle(indices)
		self.validSet = [self.trainSet[idx] for idx in indices[:numToRemove]]
		self.trainSet = [self.trainSet[idx] for idx in indices[numToRemove:]]
		print "trainSet size:",len(self.trainSet)
		print "validSet size:",len(self.validSet)

	def getPQval(self,P,Q,i,j):
		return np.dot(P[i,:],Q[:,j])

	def RMSE(self,data,P,Q):
		SSE = 0.
		for uId,iId,r in data:
			SSE += (r - self.getPQval(P,Q,uId,iId))**2
		return np.sqrt(SSE/len(data))

	def train(self,numSamples,maxEpoch,maxNumAfterBest = 2):
		"""Perform MF on self.utilityMatrix using gradient descent"""
		absolute_best_RMSE = float(1<<64-1)
		P_buffer_row = np.zeros(self.P[0,:].shape)

		for _ in xrange(numSamples):

			buff = []

			RMSE_train = float(1<<64-1)
			RMSE_valid = float(1<<64-1)
			best_RMSE_valid = float(1<<64-1)
			condition = True
			numAfterBest = 0
			bestP_valid  = np.zeros(self.P.shape)
			bestQ_valid  = np.zeros(self.Q.shape)
			P  = np.abs(np.random.randn(self.numUsers,self.numFeatures).astype(np.float64))
			Q  = np.abs(np.random.randn(self.numFeatures,self.numItems).astype(np.float64))
			epoch = 0

			#PQ = np.dot(P,Q)
			local_learningRate = self.learningRate

			try:
				while condition:
					epoch += 1
					rnd.shuffle(self.trainSet)
					for uId,iId,r in self.trainSet:
						err = r - self.getPQval(P,Q,uId,iId)
						P_buffer_row  = P[uId,:]
						P[uId,:] = P[uId,:] + local_learningRate * (err * Q[:,iId] - self.weightDecay * P[uId,:])
						Q[:,iId] = Q[:,iId] + local_learningRate * (err * P_buffer_row  - self.weightDecay * Q[:,iId])
					#PQ = np.dot(P,Q)
					oldRMSE_valid = RMSE_valid
					RMSE_train = self.RMSE(self.trainSet,P,Q)
					RMSE_valid = self.RMSE(self.validSet,P,Q)
					buff.append(oldRMSE_valid- RMSE_valid)
					
					print "\n#### Epoch",epoch,"\nRMSE_train:",RMSE_train
					print "RMSE_valid:",RMSE_valid

					if 0 < best_RMSE_valid-RMSE_valid < 0.005:
						local_learningRate *= 2.
						print "lr++"

					if best_RMSE_valid > RMSE_valid:
						best_RMSE_valid = RMSE_valid
						bestP_valid  = P
						bestQ_valid  = Q
						numAfterBest = 0
						print "(Best reconstruction: RMSE_valid), saving P and Q..."
					else:
						numAfterBest += 1

					if numAfterBest > maxNumAfterBest/2:
						local_learningRate *= 0.95
						print "lr--"
						P = bestP_valid
						Q = bestQ_valid

					if epoch > 5:
						if np.mean(buff) < 1e-3:
							local_learningRate *= 1.01
							print 'up ^'
						buff.pop()

					if numAfterBest > maxNumAfterBest or epoch > maxEpoch:
						condition = False
			except:	#This way we can interrupt our process and still save everything
				pass

			if best_RMSE_valid < absolute_best_RMSE:
				absolute_best_RMSE = best_RMSE_valid
				self.P = bestP_valid
				self.Q = bestQ_valid
				#self.PQ = np.dot(self.P,self.Q)

		print "MF over. Best RMSE:",absolute_best_RMSE

if __name__ == "__main__":

	M = np.array([[0.,2. ,4. ,4. ,3.],[3., 1., 0., 4., 1.],[2., 0., 3., 1., 4.],[2., 5., 4., 0., 0.],[4., 4., 5., 4., 0.]])
	trainSet = []
	for i in xrange(len(M)):
		for j in xrange(len(M[0,:])):
			trainSet.append((i,j,M[i,j]))
	MF = RISMF(trainSet,numItems=len(M[0,:]),numUsers=len(M),weightDecay=0.01,learningRate=0.0000001,numFeatures=5)
	MF.train(numSamples=5,maxEpoch=30000,maxNumAfterBest = 30)

	print M
	print MF.PQ
	