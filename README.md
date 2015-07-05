# rntn-spark
## Description
Repository for [MTA](https://www.mta.ac.il/en/Pages/default.aspx) final Msc. project: Distributed RNTN.  
The purpose of this project is to implement the Recurssive Neural Tensor Network (RNTN) for sentiment analysis as described in the [paper](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) by R. Socher in a distributed manner using Apache Spark.   
We are following the [Downpour](http://research.google.com/archive/large_deep_networks_nips2012.html) paradigm described by Jeffrey Dean from google and implemented by Dirk Neumann's [DeepDist](http://deepdist.com/) project.  
  
Please bare in mind: This is a work in progress! This is, by no means, a download-and-run project. 

## pre-requites and setup instructions
1. RNTN
  1. Download/clone the [semantic-rntn](https://github.com/awni/semantic-rntn) project.
  2. Make sure to run the setup.sh sccript.
  3. Change the name of the folder from "semantic-rntn" to "semantic_rntn" in order to import it as a module. For more info see [here](http://stackoverflow.com/questions/8350853/how-to-import-python-module-when-module-name-has-a-dash-or-hyphen-in-it) 

