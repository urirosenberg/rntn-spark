__author__ = 'William'
import os
import sys
import time
import optparse
import random
import numpy as np

# Path for spark source folder
#os.environ['SPARK_HOME']="/usr/local/Cellar/apache-spark/1.3.1"
os.environ["_JAVA_OPTIONS"] = "-Xmx1g"


# Append pyspark,deepdist and rntn  to Python Path
sys.path.append("/Users/William/workspace2/deepdist/deepdist")
sys.path.append("/Users/William/workspace2/semantic_rntn")
sys.path.append("/usr/local/Cellar/apache-spark/1.3.1/libexec/python/")
sys.path.append("/usr/local/Cellar/apache-spark/1.3.1/libexec/python/lib/py4j-0.8.2.1-src.zip")
#sys.path.append("/Library/Python/2.6/site-packages")
os.environ["PYTHONPATH"] += os.pathsep + "/Users/William/workspace2/deepdist/deepdist"
os.environ["PYTHONPATH"] += os.pathsep + "/Users/William/workspace2/semantic_rntn"
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
    from gensim.models.word2vec import Word2Vec
    print ("Successfully imported deepdist Modules")

except ImportError as e:
    print ("Can not import deepdist Modules", e)
    sys.exit(1)


######## import rntn ########
try:
    import rntn as nnet
    import tree as tr
    import sgd as optimizer
    print ("Successfully imported rntn Modules")

except ImportError as e:
    print ("Can not import rntn Modules", e)
    sys.exit(1)

#make sure all modules imported
time.sleep(3)

#setup args
usage = "usage : %prog [options]"
parser = optparse.OptionParser(usage=usage)
parser.add_option("--test",action="store_true",dest="test",default=False)

# Optimizer
parser.add_option("--minibatch",dest="minibatch",type="int",default=30)
parser.add_option("--optimizer",dest="optimizer",type="string",
        default="sgd")
parser.add_option("--epochs",dest="epochs",type="int",default=50)
parser.add_option("--step",dest="step",type="float",default=1e-2)
parser.add_option("--outputDim",dest="outputDim",type="int",default=5)
parser.add_option("--wvecDim",dest="wvecDim",type="int",default=30)
parser.add_option("--outFile",dest="outFile",type="string",
        default="models/test.bin")
parser.add_option("--inFile",dest="inFile",type="string",
        default="models/test.bin")
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


conf = (SparkConf()
        .setMaster("local[4]")
        .setAppName("deepdist test")
        .set("spark.executor.memory", "1g")
        .set("spark.driver.memory", "1g")
        .set("spark.python.worker.memory", "1g"))
sc = SparkContext(conf=conf)



def gradient(model, tree_data):  # executes on workers
    cost,grad = model.model.costAndGrad(tree_data)
    # compute exponentially weighted cost
    if np.isfinite(cost):
        if (model.it > 1 and  len(model.expcost) > 0):
            model.expcost.append(.01*cost + .99*model.expcost[-1])
        else:
            model.expcost.append(cost)

        if model.optimizer == 'sgd':
            update = grad
            scale = -model.alpha
    return update

def descent(model, update):      # executes on master
    scale = -model.alpha
    model.model.updateParams(scale,update,log=False)

start = time.time()
with DeepDist(sgd) as dd:
    print 'wait for server to come up'
    time.sleep(3)
    m = len(trees)
    random.shuffle(trees)
    for i in xrange(0,m-sgd.minibatch+1,sgd.minibatch):
        sgd.it += 1
        mb_data = sc.parallelize(trees[i:i+sgd.minibatch])
        dd.train(mb_data, gradient, descent)
        print '****************** finished sgd iteration %s *************************'%sgd.it
    #print dd.model.most_similar(positive=['woman', 'king'], negative=['man'])

end = time.time()
print "Time per minibatch : %f"%(end-start)
sys.exit("program done")