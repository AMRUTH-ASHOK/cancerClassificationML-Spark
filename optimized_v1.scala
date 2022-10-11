//////////////////////////////////////////
//      IMPORTING THE DEPENDENCIES      //
//////////////////////////////////////////
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{PCA,StandardScaler,VectorAssembler,StringIndexer,VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.log4j._
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator




val spark = SparkSession.builder().appName("Team5").getOrCreate()

val t1 = System.nanoTime
//////////////////////////////////////////
////          LOADING DATA            ////
//////////////////////////////////////////

val colnames = (Array(
"gene_0","gene_1","gene_2","gene_3","gene_4","gene_5","gene_6","gene_7","gene_8","gene_9","gene_10","gene_11","gene_12","gene_13","gene_14","gene_15"
))

val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("file1.csv")
// data.printSchema()
// data.show()



val df = (data.select(data("Class").as("Ilabel"),
$"gene_0",$"gene_1",$"gene_2",$"gene_3",$"gene_4",$"gene_5",$"gene_6",$"gene_7",$"gene_8",$"gene_9",$"gene_10",$"gene_11",$"gene_12",$"gene_13",$"gene_14",$"gene_15"
))

val assembler = new VectorAssembler().setInputCols(colnames).setOutputCol("features")

val output = assembler.transform(df).select($"Ilabel",$"features")



// ////////////////////////////////////////
// //        STANDARD SCALAR           ////
// ////////////////////////////////////////

val scaler = (new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(false))


val scalerModel = scaler.fit(output)
val scaledData = scalerModel.transform(output)









//////////////////////////////////////////
//     PRINCIPAL COMPONENT ANALYSIS    ///
//////////////////////////////////////////

val pca = (new PCA()
  .setInputCol("scaledFeatures")
  .setOutputCol("pcaFeatures")
  .setK(5)
  .fit(scaledData))



val AfterPCA = pca.transform(scaledData).select("Ilabel","pcaFeatures")

// AfterPCA.groupBy("Ilabel").count().show()





// //////////////////////////////////////////
// ///      CATEGORICAL TO NUMERICAL      ///
// //////////////////////////////////////////
val labelIndexer = new StringIndexer().setInputCol("Ilabel").setOutputCol("label").fit(AfterPCA)
val df4 = labelIndexer.transform(AfterPCA).select($"label", $"pcaFeatures")
val df5 = df4.select(df4("label"), df4("pcaFeatures").as("features"))





// //////////////////////////////////////////
// ///          CLASSIFICATION            ///
// //////////////////////////////////////////
val Array(training, test) = df5.randomSplit(Array(0.7, 0.3),seed=12345)

// val lr = new LogisticRegression()
val rf = new RandomForestClassifier()


val model = rf.fit(training)
val results = model.transform(test)
val duration = (System.nanoTime - t1) / 1e9d
results.show(15)
println("Time taken to run the program is "+duration+" seconds")









// //////////////////////////////////////////
// //            EVALUATION                //
// //////////////////////////////////////////

// val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
// val metrics = new MulticlassMetrics(predictionAndLabels)
// println(" ... ")
// println("Confusion matrix:" + metrics.confusionMatrix)

// println(" ")
// print("Accuracy: " + metrics.accuracy)

// println(" ")
// print("Precision: " + metrics.weightedPrecision)

// println(" ")
// print("Recall: " + metrics.weightedRecall)

