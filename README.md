# rntn-spark
## Description
Repository for [MTA](https://www.mta.ac.il/en/Pages/default.aspx) final Msc. project: Distributed RNTN.  
The purpose of this project is to implement the Recurssive Neural Tensor Network (RNTN) for sentiment analysis as described in the [paper](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) by R. Socher in a distributed manner using Apache Spark.   
We are following the [Downpour](http://research.google.com/archive/large_deep_networks_nips2012.html) paradigm described by Jeffrey Dean from google and implemented by Dirk Neumann's [DeepDist](http://deepdist.com/) project.  
  
Please bare in mind: This is a work in progress! This is, by no means, a download-and-run project. 

## pre-requites and setup instructions
1. RNTN
  1. Download/clone the [semantic-rntn](https://github.com/awni/semantic-rntn) project.
  2. Make sure to run the setup.sh script in the project.
  3. Change the name of the folder from "semantic-rntn" to "semantic_rntn" in order to import it as a module. For more info see [here](http://stackoverflow.com/questions/8350853/how-to-import-python-module-when-module-name-has-a-dash-or-hyphen-in-it) 
2. DeepDist
  1. At the moment, some updates are needed in order to run RNTN using DeepDist. Those updates are available from my forked Deepdist project. Until my pull requests are approved, Download/clone the forked [DeepDist](https://github.com/urirosenberg/deepdist) project. 
  2. Follow the setup instructions of that project, and as a sanity check make sure you are able to run the word2vec example.
3. Spark
  1. Follow the instructions on Downloading and installing Spark from the [documentation](https://spark.apache.org/docs/latest/). Make sure you know the paths to pyspark and py4j. 
4. rntn-spark
  1. Download/clone the [rntn-spark](https://github.com/urirosenberg/rntn-spark) project (this).
  2. In the configuration file: update the paths where you downloaded the RNTN, DeepDist projects and Spark's python and py4j paths.
  3. Run distrntn.py :-)

## Support
Please use github's issues to report troubles.  



