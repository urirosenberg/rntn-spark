# rntn-spark
## Description
Repository for [MTA](https://www.mta.ac.il/en/Pages/default.aspx) final Msc. project: Distributed RNTN.  
The purpose of this project is to implement the Recurssive Neural Tensor Network (RNTN) for sentiment analysis as described in the [paper](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) by R. Socher in a distributed manner using Apache Spark.   
We are following the [Downpour](http://research.google.com/archive/large_deep_networks_nips2012.html) paradigm described by Jeffrey Dean from google and implemented by Dirk Neumann's [DeepDist](http://deepdist.com/) project.  
  
Please bare in mind: This is a work in progress! This is, by no means, a download-and-run project. 

## pre-requites and setup instructions
1. RNTN
  1. Download/clone the forked [semantic-rntn](https://github.com/urirosenberg/semantic-rntn) project ***to every node on your cluster***. This is based on the original [semantic-rntn](https://github.com/awni/semantic-rntn) project. The only difference is that I have taken the existing project and turned it into a module, thus enabling it to be installed and managed on all nodes of the cluster.
  2. Install by running:  
  ```python setup.py install```
 
2. DeepDist
  1. At the moment, some updates are needed in order to run RNTN using DeepDist. Those updates are available from my forked Deepdist project. Until my pull requests are approved, Download/clone the forked [DeepDist](https://github.com/urirosenberg/deepdist) project ***to every node on your cluster***. 
  2. Install by running:  
  ```python setup.py install```
3. Spark
  1. Follow the instructions on Downloading and installing Spark from the [documentation](https://spark.apache.org/docs/latest/). Make sure you know the paths to pyspark and py4j. 
4. rntn-spark
  1. Download/clone the [rntn-spark](https://github.com/urirosenberg/rntn-spark) project (this).
  2. In the configuration file: update the paths to Spark's python and py4j paths and set the app name.
  3. Update the sparkrunner.sh script with your master address and port.
  4. Run:  
  ```sh sparkrunner.sh``

## Support
Please use github's [issues](https://github.com/urirosenberg/rntn-spark/issues) to report troubles.  



