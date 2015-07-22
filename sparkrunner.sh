#!/usr/bin/env bash
export JAVA_HOME=/usr/java/default/jre/
export SPARK_HOME=/opt/cloudera/parcels/CDH-5.3.1-1.cdh5.3.1.p0.5/lib/spark
export PYTHONPATH=/usr/bin/python
spark-submit --master spark://n12.mta.ac.il:7077  distrntn/distrntn.py
#spark-submit --master spark://n12.mta.ac.il:7077  --total-executor-cores 1 distrntn/distrntn.py
