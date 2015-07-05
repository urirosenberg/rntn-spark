__author__ = 'William'
import os
import sys
import time
import optparse
import cPickle as pickle
import cloudpickle as pickle2
import ConfigParser


config = ConfigParser.ConfigParser()
config.readfp(open(r'config'))
path1 = config.get('My Section', 'path1')


'''
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
#trees = tr.loadTrees()
opts.numWords = 2


#setup the rntn
rnn = nnet.RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
rnn.initParams()
t1=rnn.stack
t2=rnn.stack
for P,dP in zip(t1[1:],t2[1:]):
        print "******* descent test P: %s *******"%str(P)
        print "******* descent test dP: %s *******"%str(dP)


sgd = optimizer.SGD(rnn,alpha=opts.step,minibatch=opts.minibatch,
    optimizer=opts.optimizer)




pmodel = pickle2.dumps(sgd.model, -1)
pmodel2 = pickle.loads(pmodel)
'''
sys.exit("program done")