import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, OneHotEncoderEstimator}
import org.apache.spark.ml.linalg.Vectors
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
    setOutputCol("countryIndex")
val indexed1 = indexer1.fit(data).transform(data)

val indexer2 = new StringIndexer().
    setInputCol("gender").
    setOutputCol("genderIndex")
val indexed2 = indexer2.fit(indexed1).transform(indexed1)

// one hot encoding
val encoder = new OneHotEncoderEstimator().
  setInputCols(Array("countryIndex", "genderIndex")).
  setOutputCols(Array("countryVec", "genderVec"))
val encoded = encoder.fit(indexed2).transform(indexed2)

// filter out null columns
val dropped1 = encoded.filter("YoB != 'NA'")

// check null values for :
//last_event_DI, nevents, ndays_act, nplay_video, nchapters, nforum_posts,


/*
// Set the input columns from which we are supposed to read the values
// Set the name of the column where the vector will be stored
val assembler = new VectorAssembler().setInputCols(Array("Avg Area Income","Avg Area House Age","Avg Area Number of Rooms","Area Population")).setOutputCol("features")

// Transform the DataFrame
val output = assembler.transform(df).select($"label",$"features")

// Create an array of the training and test data
val Array(training, test) = output.select("label","features").randomSplit(Array(0.7, 0.3), seed = 12345)

//////////////////////////////////////
//////// Random Forest Classifier //////////
////////////////////////////////////
val rf = new RandomForestClassifier()

//////////////////////////////////////
/// PARAMETER GRID BUILDER //////////
////////////////////////////////////
val paramGrid = new ParamGridBuilder().addGrid().build()

///////////////////////
// TRAIN TEST SPLIT //
/////////////////////

// In this case the estimator is simply the linear regression.
// A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
// 80% of the data will be used for training and the remaining 20% for validation.
val trainValidationSplit = (new TrainValidationSplit()
                            .setEstimator(lr)
                            .setEvaluator(new RegressionEvaluator.setMetricName("r2") )
                            .setEstimatorParamMaps(paramGrid)
                            .setTrainRatio(0.8) )


// You can then treat this object as the new model and use fit on it.
// Run train validation split, and choose the best set of parameters.
val model = trainValidationSplit.fit(training)

//////////////////////////////////////
// EVALUATION USING THE TEST DATA ///
////////////////////////////////////

// Make predictions on test data. model is the model with combination of parameters
// that performed best.
model.transform(test).select("features", "label", "prediction").show()

// Check out the metrics
model.validationMetrics

*/