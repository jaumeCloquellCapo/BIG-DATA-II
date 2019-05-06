package main.scala.djgarcia

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.{SparkConf, SparkContext}

object runEnsembles {

  def main(arg: Array[String]) {

    //Basic setup
    val jobName = "MLlib Ensembles"

    //Spark Configuration
    val conf = new SparkConf().setAppName(jobName)
    val sc = new SparkContext(conf)

    //Load train and test
    val pathTrain = "file:///home/spark/datasets/susy-10k-tra.data"
    val rawDataTrain = sc.textFile(pathTrain)

    val pathTest = "file:///home/spark/datasets/susy-10k-tst.data"
    val rawDataTest = sc.textFile(pathTest)

    val train = rawDataTrain.map { line =>
      val array = line.split(",")
      val arrayDouble = array.map(f => f.toDouble)
      val featureVector = Vectors.dense(arrayDouble.init)
      val label = arrayDouble.last
      LabeledPoint(label, featureVector)
    }.persist

    train.count
    train.first

    val test = rawDataTest.map { line =>
      val array = line.split(",")
      val arrayDouble = array.map(f => f.toDouble)
      val featureVector = Vectors.dense(arrayDouble.init)
      val label = arrayDouble.last
      LabeledPoint(label, featureVector)
    }.persist

    test.count
    test.first

    //Load train and test with KeelParser

    /*val converter = new KeelParser(sc, "file:///home/spark/datasets/susy.header")
    val train = sc.textFile("file:///home/spark/datasets/susy-10k-tra.data", 10).map(line => converter.parserToLabeledPoint(line)).persist
    val test = sc.textFile("file:///home/spark/datasets/susy-10k-tst.data", 10).map(line => converter.parserToLabeledPoint(line)).persist
*/

    //Class balance

    val classInfo = train.map(lp => (lp.label, 1L)).reduceByKey(_ + _).collectAsMap()


    //Decision tree

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    var numClasses = 2
    var categoricalFeaturesInfo = Map[Int, Int]()
    var impurity = "gini"
    var maxDepth = 5
    var maxBins = 32

    val modelDT = DecisionTree.trainClassifier(train, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPredsDT = test.map { point =>
      val prediction = modelDT.predict(point.features)
      (point.label, prediction)
    }
    val testAccDT = 1 - labelAndPredsDT.filter(r => r._1 != r._2).count().toDouble / test.count()
    println(s"Test Accuracy DT= $testAccDT")


    //Random Forest

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    numClasses = 2
    categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 100
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    impurity = "gini"
    maxDepth = 4
    maxBins = 32

    val modelRF = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPredsRF = test.map { point =>
      val prediction = modelRF.predict(point.features)
      (point.label, prediction)
    }
    val testAccRF = 1 - labelAndPredsRF.filter(r => r._1 != r._2).count.toDouble / test.count()
    println(s"Test Accuracy RF= $testAccRF")


    //PCARD

    import org.apache.spark.mllib.tree.PCARD

    val cuts = 5
    val trees = 10

    val pcardTrain = PCARD.train(train, trees, cuts)

    val pcard = pcardTrain.predict(test)

    val labels = test.map(_.label).collect()

    var cont = 0

    for (i <- labels.indices) {
      if (labels(i) == pcard(i)) {
        cont += 1
      }
    }

    val testAcc = cont / labels.length.toFloat

    println(s"Test Accuracy = $testAcc")


    val predsAndLabels = sc.parallelize(pcard).zipWithIndex.map { case (v, k) => (k, v) }.join(test.zipWithIndex.map { case (v, k) => (k, v.label) }).map(_._2)

    //Metrics

    import org.apache.spark.mllib.evaluation.MulticlassMetrics

    val metrics = new MulticlassMetrics(predsAndLabels)
    val precision = metrics.precision
    val cm = metrics.confusionMatrix

    //Write Results
    /*val writer = new PrintWriter("/home/cdXX/results.txt")
    writer.write(
      "Precision: " + precision + "\n" +
        "Confusion Matrix " + cm + "\n"
    )
    writer.close()*/
  }
}
