from pyspark.mllib.feature import HashingTF, IDF

from pyspark.mllib.classification import LogisticRegressionModel

from pyspark.mllib.classification import LogisticRegressionWithSGD

from pyspark.mllib.regression import LabeledPoint
import sys

from pyspark.sql import SparkSession
from pyspark import SparkContext

from pyspark import SparkConf
from pyspark.context import SparkContext

from pyspark.mllib.recommendation import MatrixFactorizationModel

import os

if __name__ == '__main__':
    
    print ("This is the name of the script: ", sys.argv[0])
    print ("Number of arguments: ", len(sys.argv))
    print ("The arguments are: " , str(sys.argv))
    
    spamTrainingInputPath = sys.argv[1]
    nonSpamTrainingInputPath = sys.argv[2]
    builtModelPath = os.path.dirname(sys.argv[3])
    conf = SparkConf()
    conf.setAppName("SpamDetectionModelBuilder")
    sc = SparkContext.getOrCreate(conf=conf)
    spam = sc.textFile(spamTrainingInputPath)
    nonspam =sc.textFile(nonSpamTrainingInputPath)

    spamw = spam.map(lambda x: x.split())
    nonspamw = nonspam.map(lambda x: x.split())

    tf = HashingTF(numFeatures = 1000)

    def createdLabeledPoint(label, data, tf) :

        data = data.map(lambda x: x.split())
        datatf = tf.transform(data)
        result = datatf.map(lambda x:LabeledPoint(label, x))
        return result

    spamresult = createdLabeledPoint(1, spam, tf)
    nonspamresult = createdLabeledPoint(0, nonspam, tf)

    trainingData = spamresult.union(nonspamresult)

    model = LogisticRegressionWithSGD.train(trainingData)

    model.save(sc, builtModelPath)