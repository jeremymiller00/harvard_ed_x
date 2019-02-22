import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, CrossValidator}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, OneHotEncoderEstimator}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Start a simple Spark Session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

// Prepare training and test data.
val data = spark.read.option("header","true").
    option("inferSchema","true").
    format("csv").
    load("/Users/jeremymiller/GoogleDrive/Data_Science/Projects/Education_Data/harvard_ed_x/data/mooc.csv")

///////////////////////////////////////////////////
//// Setting Up DataFrame for Machine Learning ////
///////////////////////////////////////////////////

// Rename label column, keep appropriate columns
val df = data.select(data("certified").as("label"), $"registered", $"viewed", $"explored", $"final_cc_cname_DI", $"gender", $"nevents", $"ndays_act", $"nplay_video", $"nchapters", $"nforum_posts")

// string indexing
val indexer1 = new StringIndexer().
    setInputCol("final_cc_cname_DI").
    setOutputCol("countryIndex").
    setHandleInvalid("keep") 
val indexed1 = indexer1.fit(df).transform(df)

val indexer2 = new StringIndexer().
    setInputCol("gender").
    setOutputCol("genderIndex").
    setHandleInvalid("keep")
val indexed2 = indexer2.fit(indexed1).transform(indexed1)

// one hot encoding
val encoder = new OneHotEncoderEstimator().
  setInputCols(Array("countryIndex", "genderIndex")).
  setOutputCols(Array("countryVec", "genderVec"))
val encoded = encoder.fit(indexed2).transform(indexed2)

// filter out null columns
val dropped1 = encoded.filter("YoB != 'NA'")

// check null values for : impute with median
//nevents 174059, ndays_act 145178, nplay_video 395025, nchapters 224464,

// define medians
val neventsMedianArray = dropped1.stat.approxQuantile("nevents", Array(0.5), 0)
val neventsMedian = neventsMedianArray(0)

val ndays_actMedianArray = dropped1.stat.approxQuantile("ndays_act", Array(0.5), 0)
val ndays_actMedian = ndays_actMedianArray(0)

val nplay_videoMedianArray = dropped1.stat.approxQuantile("nplay_video", Array(0.5), 0)
val nplay_videoMedian = nplay_videoMedianArray(0)

val nchaptersMedianArray = dropped1.stat.approxQuantile("nchapters", Array(0.5), 0)
val nchaptersMedian = nchaptersMedianArray(0)

// replace 
val filled = dropped1.na.fill(Map(
  "nevents" -> neventsMedian, 
  "ndays_act" -> ndays_actMedian, 
  "nplay_video" -> nplay_videoMedian, 
  "nchapters" -> nchaptersMedian))


// Set the input columns from which we are supposed to read the values
// Set the name of the column where the vector will be stored
val assembler = new VectorAssembler().setInputCols(Array("viewed", "explored", "nevents", "ndays_act", "nplay_video", "nchapters", "nforum_posts", "countryVec", "genderVec")).setOutputCol("features")

// Transform the DataFrame
val output = assembler.transform(filled).select($"label",$"features")

// Splitting the data by create an array of the training and test data
val Array(training, test) = output.select("label","features").randomSplit(Array(0.7, 0.3), seed = 12345)

//////////////////////////////////////
//////// Random Forest Classifier //////////
////////////////////////////////////
val rf = new RandomForestClassifier()

//////////////////////////////////////
/// PARAMETER GRID BUILDER //////////
////////////////////////////////////
val paramGrid = new ParamGridBuilder().
  addGrid(rf.numTrees,Array(20,50,100)).
  build()

///////////////////////
// Cross Validation //
/////////////////////

val cv = new CrossValidator().
  setEstimator(rf).
  setEvaluator(new MulticlassClassificationEvaluator().setMetricName("weightedRecall")).
  setEstimatorParamMaps(paramGrid).
  setNumFolds(3).
  setParallelism(2)


// You can then treat this object as the new model and use fit on it.
// Run train validation split, and choose the best set of parameters.
val model = cv.fit(training)

//////////////////////////////////////
// EVALUATION USING THE TEST DATA ///
////////////////////////////////////

// Make predictions on test data. model is the model with combination of parameters
// that performed best.
// val predictions = model.transform(test)

val results = model.transform(test).select("features", "label", "prediction")

//this doesn't work because the metrics in Spark all return only the accuracy
// val evaluator = new MulticlassClassificationEvaluator().
//   setLabelCol("label").
//   setPredictionCol("prediction").
//   setMetricName("accuracy")

// val testRecall = evaluator.evaluate(testPreds)

val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

// Instantiate a new metrics objects
val bMetrics = new BinaryClassificationMetrics(predictionAndLabels)
val mMetrics = new MulticlassMetrics(predictionAndLabels)
val labels = mMetrics.labels

// Print out the Confusion matrix
println("Confusion matrix:")
println(mMetrics.confusionMatrix)

// Precision by label
labels.foreach { l =>
  println(s"Precision($l) = " + mMetrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + mMetrics.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + mMetrics.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + mMetrics.fMeasure(l))
}


// Precision by threshold
val precision = bMetrics.precisionByThreshold
precision.foreach { case (t, p) =>
  println(s"Threshold: $t, Precision: $p")
}

// Recall by threshold
val recall = bMetrics.recallByThreshold
recall.foreach { case (t, r) =>
  println(s"Threshold: $t, Recall: $r")
}

// Precision-Recall Curve
val PRC = bMetrics.pr

// F-measure
val f1Score = bMetrics.fMeasureByThreshold
f1Score.foreach { case (t, f) =>
  println(s"Threshold: $t, F-score: $f, Beta = 1")
}

val beta = 0.5
val fScore = bMetrics.fMeasureByThreshold(beta)
f1Score.foreach { case (t, f) =>
  println(s"Threshold: $t, F-score: $f, Beta = 0.5")
}

// AUPRC
val auPRC = bMetrics.areaUnderPR
println("Area under precision-recall curve = " + auPRC)

// Compute thresholds used in ROC and PR curves
val thresholds = precision.map(_._1)

// ROC Curve
val roc = bMetrics.roc

// AUROC
val auROC = bMetrics.areaUnderROC
println("Area under ROC = " + auROC)
