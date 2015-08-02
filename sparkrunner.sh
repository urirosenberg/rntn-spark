#!/usr/bin/env bash
export JAVA_HOME=/opt/jdk1.7.0_71/jre/
export SPARK_HOME=/usr/lib/spark-1.3.1-bin-hadoop2.6
export PYTHONPATH=/usr/bin/python
spark-submit --master spark://<master-url>:<master-port>  distrntn/distrntn.py
#spark-submit --master spark://<master-url>:<master-port>  --total-executor-cores 1 distrntn/distrntn.py
