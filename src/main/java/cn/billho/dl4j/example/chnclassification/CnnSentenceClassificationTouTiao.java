package cn.billho.dl4j.example.chnclassification;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.common.Term;
import lombok.SneakyThrows;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.arbiter.util.ClassPathResource;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.*;

/**
 * 对头条数据进行分类
 * //ref： https://blog.csdn.net/chutuo1904/article/details/100683975?depth_1-utm_source=distribute.pc_relevant_right.none-task&utm_source=distribute.pc_relevant_right.none-task
 * dataset： https://github.com/fate233/toutiao-text-classfication-dataset
 */
public class CnnSentenceClassificationTouTiao {

    private static String dataDir = "dataset/toutiao-text-classfication-dataset/";
    public static void main(String[] args) throws Exception {
//        preProcess();
//        trainWord2Vec();
        trainCNNModel();
    }

    /**
     * 训练
     * @throws IOException
     */
    private static void trainCNNModel() throws IOException {
        List<String> trainLabelList = new ArrayList<>();// 训练集label
        List<String> trainSentences = new ArrayList<>();// 训练集文本集合
        List<String> testLabelList = new ArrayList<>();// 测试集label
        List<String> testSentences = new ArrayList<>();//// 测试集文本集合
        Map<String, List<String>> map = new HashMap<>();

        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(
                new FileInputStream(new File(dataDir + "toutiao_data_type_word.txt")), "UTF-8"));
        String line = null;
        int truncateReviewsToLength = 0;
        Random random = new Random(123);
        while ((line = bufferedReader.readLine()) != null) {
            String[] array = line.split("_!_");
            if (map.get(array[0]) == null) {
                map.put(array[0], new ArrayList<String>());
            }
            map.get(array[0]).add(array[1]);// 将样本中所有数据，按照类别归类
            int length = array[1].split(" ").length;
            if (length > truncateReviewsToLength) {
                truncateReviewsToLength = length;// 求样本中，句子的最大长度
            }
        }
        bufferedReader.close();
        for (Map.Entry<String, List<String>> entry : map.entrySet()) {
            for (String sentence : entry.getValue()) {
                if (random.nextInt() % 5 == 0) {// 每个类别抽取20%作为test集
                    testLabelList.add(entry.getKey());
                    testSentences.add(sentence);
                } else {
                    trainLabelList.add(entry.getKey());
                    trainSentences.add(sentence);
                }
            }

        }
        int batchSize = 64;
        int vectorSize = 100;
        int nEpochs = 3;
//        int nEpochs = 10;

        int cnnLayerFeatureMaps = 30;
//        int cnnLayerFeatureMaps = 50;
        PoolingType globalPoolingType = PoolingType.MAX;
        Random rng = new Random(12345);
        Nd4j.getMemoryManager().setAutoGcWindow(5000);
//        truncateReviewsToLength = 50;
        System.out.println("truncateReviewsToLength:" + truncateReviewsToLength);

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder().weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU).updater(new Nesterovs(0.01, 0.9))
                .convolutionMode(ConvolutionMode.Same).l2(0.0001).graphBuilder().addInputs("input")
                .addLayer("cnn3",
                        //kernelSize(3, vectorSize)： height * weight
                        new ConvolutionLayer.Builder().kernelSize(3, vectorSize).stride(1, vectorSize) // vector：100
                                .nOut(cnnLayerFeatureMaps).build(),//cnnLayerFeatureMaps：50
                        "input")
                // 参数个数计算：IN * kernelSize(3, vectorSize)： height * weight * out（cnnLayerFeatures）(特征数)，有多少特征，就有多少输出。
                .addLayer("cnn4",
                        new ConvolutionLayer.Builder().kernelSize(4, vectorSize).stride(1, vectorSize)
                                .nOut(cnnLayerFeatureMaps).build(),
                        "input")
                .addLayer("cnn5",
                        new ConvolutionLayer.Builder().kernelSize(5, vectorSize).stride(1, vectorSize)
                                .nOut(cnnLayerFeatureMaps).build(),
                        "input")
                .addLayer("cnn6",
                        new ConvolutionLayer.Builder().kernelSize(6, vectorSize).stride(1, vectorSize)
                                .nOut(cnnLayerFeatureMaps).build(),
                        "input")
                .addLayer("cnn3-stride2",
                        new ConvolutionLayer.Builder().kernelSize(3, vectorSize).stride(2, vectorSize)
                                .nOut(cnnLayerFeatureMaps).build(),
                        "input")
                .addLayer("cnn4-stride2",
                        new ConvolutionLayer.Builder().kernelSize(4, vectorSize).stride(2, vectorSize)
                                .nOut(cnnLayerFeatureMaps).build(),
                        "input")
                .addLayer("cnn5-stride2",
                        new ConvolutionLayer.Builder().kernelSize(5, vectorSize).stride(2, vectorSize)
                                .nOut(cnnLayerFeatureMaps).build(),
                        "input")
                .addLayer("cnn6-stride2",
                        new ConvolutionLayer.Builder().kernelSize(6, vectorSize).stride(2, vectorSize)
                                .nOut(cnnLayerFeatureMaps).build(),
                        "input")
                .addVertex("merge1", new MergeVertex(), "cnn3", "cnn4", "cnn5", "cnn6")
                .addLayer("globalPool1", new GlobalPoolingLayer.Builder().poolingType(globalPoolingType).build(),
                        "merge1")
                .addVertex("merge2", new MergeVertex(), "cnn3-stride2", "cnn4-stride2", "cnn5-stride2", "cnn6-stride2")
                .addLayer("globalPool2", new GlobalPoolingLayer.Builder().poolingType(globalPoolingType).build(),
                        "merge2")
                .addLayer("fc",
                        new DenseLayer.Builder().nOut(200).dropOut(0.5).activation(Activation.LEAKYRELU).build(),
                        "globalPool1", "globalPool2")
                .addLayer("out",
                        new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX).nOut(15).build(),
                        "fc")
                .setOutputs("out").setInputTypes(InputType.convolutional(truncateReviewsToLength, vectorSize, 1))
                .build();

        ComputationGraph net = new ComputationGraph(config);
        net.init();
        System.out.println(net.summary());
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(dataDir + "toutiao.vec");
        System.out.println("Loading word vectors and creating DataSetIterators");
        DataSetIterator trainIter = getDataSetIterator(word2Vec, batchSize, truncateReviewsToLength, trainLabelList,
                trainSentences, rng);
        DataSetIterator testIter = getDataSetIterator(word2Vec, batchSize, truncateReviewsToLength, testLabelList,
                testSentences, rng);

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        net.setListeners(new ScoreIterationListener(100), new StatsListener(statsStorage, 20),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));

        // net.setListeners(new ScoreIterationListener(100),
        // new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));
        net.fit(trainIter, nEpochs);
    }

    private static DataSetIterator getDataSetIterator(WordVectors wordVectors, int minibatchSize, int maxSentenceLength,
                                                      List<String> lableList, List<String> sentences, Random rng) {

        LabeledSentenceProvider sentenceProvider = new CollectionLabeledSentenceProvider(sentences, lableList, rng);

        return new CnnSentenceDataSetIterator.Builder().sentenceProvider(sentenceProvider).wordVectors(wordVectors)
                .minibatchSize(minibatchSize).maxSentenceLength(maxSentenceLength).useNormalizedWordVectors(false)
                .build();
    }


    /**
     * 训练Word2Vec
     */
    public static void trainWord2Vec(){
        try {
            String filePath = /*new ClassPathResource*/(dataDir + "toutiao_data_word.txt")
                    /*.getFile().getAbsolutePath()*/;
            SentenceIterator iter = new BasicLineIterator(filePath);
            TokenizerFactory t = new DefaultTokenizerFactory();
            t.setTokenPreProcessor(new CommonPreprocessor());
            VocabCache<VocabWord> cache = new AbstractCache<VocabWord>();
            WeightLookupTable<VocabWord> table = new InMemoryLookupTable.Builder<VocabWord>().vectorLength(100)
                    .useAdaGrad(false).cache(cache).build();

            //log.info("Building model....");
            Word2Vec vec = new Word2Vec.Builder()
                    .elementsLearningAlgorithm("org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW")
                    .minWordFrequency(0).iterations(1).epochs(20).layerSize(100).seed(42).windowSize(8).iterate(iter)
                    .tokenizerFactory(t).lookupTable(table).vocabCache(cache).build();

            vec.fit();
            WordVectorSerializer.writeWord2VecModel(vec, dataDir + "toutiao.vec");
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    /**
     * 数据预处理
     */
    @SneakyThrows
    public static void preProcess(){
        File dir = new File(dataDir);
        if (!dir.exists()){
            dir.mkdirs();
        }
        String catDataPath = new ClassPathResource("dataset/toutiao/toutiao_cat_data.txt").getFile().getAbsolutePath();
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(
                new FileInputStream(new File(catDataPath)), "UTF-8"));
        OutputStreamWriter writerStream = new OutputStreamWriter(
                new FileOutputStream(dataDir + "toutiao_data_type_word.txt"), "UTF-8");
        BufferedWriter writer = new BufferedWriter(writerStream);
        OutputStreamWriter writerStream2 = new OutputStreamWriter(
                new FileOutputStream(dataDir + "toutiao_data_word.txt"), "UTF-8");
        BufferedWriter writer2 = new BufferedWriter(writerStream2);
        String line = null;
        long startTime = System.currentTimeMillis();
        while ((line = bufferedReader.readLine()) != null) {
            String[] array = line.split("_!_");
            StringBuilder stringBuilder = new StringBuilder();
            for (Term term : HanLP.segment(array[3])) {
                if (stringBuilder.length() > 0) {
                    stringBuilder.append(" ");
                }
                stringBuilder.append(term.word.trim());
            }
            writer.write(Integer.parseInt(array[1].trim()) + "_!_" + stringBuilder.toString() + "\n");
            writer2.write(stringBuilder.toString() + "\n");
        }
        writer.flush();
        writer.close();
        writer2.flush();
        writer2.close();
        System.out.println(System.currentTimeMillis() - startTime);
        bufferedReader.close();
    }
}

