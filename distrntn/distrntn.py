__author__ = 'Uri'
import os
import sys
import time
import optparse
import random
import numpy as np
import ConfigParser
import socket
import pickle
import collections
print ("Running on %s"%socket.gethostname())

#Load config parameters
config = ConfigParser.ConfigParser()
config.readfp(open(r'../config'))
SparkPythonPath = config.get('Spark', 'SparkPythonPath')
Py4jPath = config.get('Spark', 'Py4jPath')
appname = config.get('distrntn', 'appname')
masterurl = config.get('distrntn', 'masterurl')
mode = config.get('distrntn', 'mode')

sys.path.append(SparkPythonPath)
sys.path.append(Py4jPath)
print os.environ["PYTHONPATH"]



######## import spark ########
try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

######## import deepdist ########
try:
    from deepdist import DeepDist
    print ("Successfully imported deepdist Modules")

except ImportError as e:
    print ("Can not import deepdist Modules", e)
    sys.exit(1)


######## import rntn ########
try:
    from semantic_rntn import rntn as nnet
    from semantic_rntn import tree as tr
    from semantic_rntn import sgd as optimizer
    print ("Successfully imported rntn Modules")

except ImportError as e:
    print ("Can not import rntn Modules", e)
    sys.exit(1)

#make sure all modules imported
time.sleep(8)

'''
Setup args for sgd.
Note: Since Deepdist implements stochastic gradient descent the model type (optimizer) has to be sgd.
'''
usage = "usage : %prog [options]"
parser = optparse.OptionParser(usage=usage)
parser.add_option("--test",action="store_true",dest="test",default=False)

# Optimizer
parser.add_option("--minibatch",dest="minibatch",type="int",default=90)
parser.add_option("--optimizer",dest="optimizer",type="string",
        default="sgd")
parser.add_option("--epochs",dest="epochs",type="int",default=50)
parser.add_option("--step",dest="step",type="float",default=1e-2)
parser.add_option("--outputDim",dest="outputDim",type="int",default=5)
parser.add_option("--wvecDim",dest="wvecDim",type="int",default=30)
parser.add_option("--outFile",dest="outFile",type="string",
        default="models/distrntn.bin")
parser.add_option("--inFile",dest="inFile",type="string",
        default="models/distrntn.bin")
parser.add_option("--data",dest="data",type="string",default="train")
(opts,args)=parser.parse_args(None)


print "Loading data..."
# load training data
trees = tr.loadTrees()
opts.numWords = len(tr.loadWordMap())


#setup the rntn
rnn = nnet.RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
rnn.initParams()
sgd = optimizer.SGD(rnn,alpha=opts.step,minibatch=opts.minibatch,
    optimizer=opts.optimizer)

#setup spark
if mode == "local":
   # Set heap space size for java
   #os.environ["_JAVA_OPTIONS"] = "-Xmx1g"
   conf = (SparkConf()
           .setMaster("local[*]")
           .setAppName(appname)
           .set("spark.executor.memory", "1g")
           .set("spark.driver.memory", "1g")
           .set("spark.python.worker.memory", "1g"))

if mode == "cluster":
   conf = (SparkConf()
           .setAppName(appname))

sc = SparkContext(conf=conf)


'''
Define the gradient and descent functions as required by DeepDist.
For more info about gradient and descent functions, please see: http://www.deepdist.com
'''
def gradient(model, tree_data):  # executes on workers
    """
    Each datum in the minibatch is a tree.
    Forward prop each tree.
    Backprop each tree.
    Returns
       cost
       Gradient w.r.t. W, Ws, b, bs
       Gradient w.r.t. L in sparse form.
    """
    tree_data=list(tree_data)
    datasize=len(tree_data)
    print "-----Running gradient. data size is %s of type %s-------"%(datasize,type(tree_data))
    if datasize == 0:
        return []
    cost = 0.0
    correct = 0.0
    total = 0.0



    model.model.L,model.model.V,model.model.W,model.model.b,model.model.Ws,model.model.bs = model.model.stack
    # Zero gradients
    model.model.dV[:] = 0
    model.model.dW[:] = 0
    model.model.db[:] = 0
    model.model.dWs[:] = 0
    model.model.dbs[:] = 0
    model.model.dL = collections.defaultdict(model.model.defaultVec)


    # Forward prop each tree in minibatch
    for tree in tree_data:
        c,corr,tot =  model.model.forwardProp(tree.root)
        cost += c
        correct += corr
        total += tot

    # Back prop each tree in minibatch
    for tree in tree_data:
        model.model.backProp(tree.root)

    # scale cost and grad by mb size ************************88
    #scale = (1./model.model.mbSize)
    scale = (1./datasize)
    for v in model.model.dL.itervalues():
        v *=scale

    # Add L2 Regularization
    cost += (model.model.rho/2)*np.sum(model.model.V**2)
    cost += (model.model.rho/2)*np.sum(model.model.W**2)
    cost += (model.model.rho/2)*np.sum(model.model.Ws**2)

    return [scale*cost,[model.model.dL,scale*(model.model.dV+model.model.rho*model.model.V),
                       scale*(model.model.dW + model.model.rho*model.model.W),scale*model.model.db,
                       scale*(model.model.dWs+model.model.rho*model.model.Ws),scale*model.model.dbs]]

def descent(model, update):      # executes on master
    if len(update) != 0:
        cost=update[0]
        print "***************** cost: %s"%cost
        grad=update[1]
        scale = -model.alpha
        # compute exponentially weighted cost
        if np.isfinite(cost):
            if (model.it > 1 and  len(model.expcost) > 0):
                model.expcost.append(.01*cost + .99*model.expcost[-1])
            else:
                model.expcost.append(cost)

            if model.optimizer == 'sgd':
                update = grad
                scale = -model.alpha
        model.model.updateParams(scale,update,log=True)

start = time.time()
with DeepDist(sgd,masterurl) as dd:
    print 'wait for server to come up'
    time.sleep(10)
    #epoch loop
    for e in range(opts.epochs):
        startepoch = time.time()
        print "Running epoch %d"%e
        m = len(trees)
        random.shuffle(trees)
        for i in xrange(0,m-sgd.minibatch+1,sgd.minibatch):
            sgd.it += 1
            mb_data = sc.parallelize(trees[i:i+sgd.minibatch])
            dd.train(mb_data, gradient, descent)
        endepoch= time.time()
        print '******** time of iteration %f'%(endepoch-startepoch)

end = time.time()
print "Total time: %f"%(end-start)

#output the final model to file
with open(opts.outFile,'w') as fid:
   pickle.dump(opts,fid)
   pickle.dump(sgd.costt,fid)
   rnn.toFile(fid)

sys.exit("program done")
