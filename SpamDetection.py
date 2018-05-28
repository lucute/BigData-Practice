
from pyspark import SparkContext
from pyspark.context import SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import LogisticRegressionModel
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
import sys
from pyspark import SparkConf

if __name__ == '__main__':

	print("This is the name of the script: ", sys.argv[0])
	print("Number of arguments: ", len(sys.argv))
	print("The arguments are: " , str(sys.argv))

	queryInputPath = sys.argv[1]
	savedModelPath = sys.argv[2]

	conf = SparkConf()
	conf.setAppName("SpamDetection")
	sc = SparkContext.getOrCreate(conf=conf)

	model = LogisticRegressionModel.load(sc, savedModelPath)

	query = sc.textFile(queryInputPath, use_unicode=False)

	tf = HashingTF(numFeatures = 1000)

	def classify(data):
		data2 = data.split()
		datatf = tf.transform(data2)
		classifications = model.predict(datatf)

		return classifications

	classifications = query.map(lambda x: (classify(x), x))

	predictions = classifications.collect()
	spam = 0
	nonspam = 0

	for x in predictions:
		if x[0] == 0:
			nonspam += 1
		elif x[0] == 1:
			spam += 1

		print("prediction = " + str(x[0]))
		print("query email = " + str(x[1]))

	print("Nonspam: " + str(nonspam))
	print("spam: " + str(spam))

	spark.stop()