__author__ = 'William'
import os
import sys
import time
import optparse
import cPickle as pickle
import cloudpickle as pickle2
import ConfigParser

'''
config = ConfigParser.ConfigParser()
config.readfp(open(r'config'))
path1 = config.get('My Section', 'path1')
'''

'''
# Path for spark source folder
#os.environ['SPARK_HOME']="/usr/local/Cellar/apache-spark/1.3.1"
os.environ["_JAVA_OPTIONS"] = "-Xmx1g"
'''

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



#setup spark
#.setMaster("spark://127.0.0.1:7077")
conf = (SparkConf()
        .setMaster("spark://127.0.0.1:7077")
        .setAppName("test")
        .set("spark.executor.memory", "1g")
        .set("spark.driver.memory", "1g")
        .set("spark.python.worker.memory", "1g"))
sc = SparkContext(conf=conf)

words = sc.textFile("test.txt")
words.filter(lambda w: w.startswith("spar")).take(5)
sys.exit("program done")