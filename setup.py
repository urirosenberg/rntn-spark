__author__ = 'William'
import os
import sys

# Path for spark source folder
#os.environ['SPARK_HOME']="/usr/local/Cellar/apache-spark/1.3.1"
os.environ["_JAVA_OPTIONS"] = "-Xmx1g"


# Append pyspark  to Python Path
sys.path.append("/usr/local/Cellar/apache-spark/1.3.1/libexec/python/")
sys.path.append("/usr/local/Cellar/apache-spark/1.3.1/libexec/python/lib/py4j-0.8.2.1-src.zip")
sys.path.append("/Users/William/workspace2/deepdist/deepdist")
os.environ["PYTHONPATH"] += os.pathsep + "/Users/William/workspace2/deepdist/deepdist"

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

try:
    from deepdist import DeepDist
    from gensim.models.word2vec import Word2Vec
    print ("Successfully imported deepdist Modules")

except ImportError as e:
    print ("Can not import deepdist Modules", e)
    sys.exit(1)

conf = (SparkConf()
        .setMaster("local[*]")
        .setAppName("deepdist test")
        .set("spark.executor.memory", "1g")
        .set("spark.driver.memory", "1g")
        .set("spark.python.worker.memory", "1g"))
sc = SparkContext(conf=conf)

corpus = sc.textFile('xaa').map(lambda s: s.split())

def gradient(model, sentences):  # executes on workers
    syn0, syn1 = model.syn0.copy(), model.syn1.copy()
    model.train(sentences)
    return {'syn0': model.syn0 - syn0, 'syn1': model.syn1 - syn1}

def descent(model, update):      # executes on master
    model.syn0 += update['syn0']
    model.syn1 += update['syn1']

with DeepDist(Word2Vec(corpus.collect())) as dd:
    dd.train(corpus, gradient, descent)
    print dd.model.most_similar(positive=['woman', 'king'], negative=['man'])

sys.exit("program done")