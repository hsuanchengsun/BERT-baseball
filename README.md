#  Using BERT on Prediction Problems with Numeric Input Data: the Case of Major League Baseball Home Run Prediction

#### Github version 1

##### Hsuan-Cheng Sun
##### Department of Mathematical Sciences National Chengchi University
##### 107751002@g.nccu.edu.tw

### Abstract

BERT is a powerful deep learning model in nature language processing. It performs well in various language tasks such as machine translation and question answering since it has great ability to analyze word sequence. In this paper, we show that BERT is able to make prediction with numerical data input instead of text. We want to predict output with numerical data and verify its performance. In particular, we choose the home run performance prediction task which input the stats of players in Major League Baseball. We also compare result of BERT-based approach with the performance of LSTM-based model and the popular projection system ZiPS. In testing data of year 2018, Bert-based approach reaches 50.6% accuracy while LSTM-based model has 48.8% and ZiPS gets only 25.4% accuracy rate. In 2019, BERT achieves 44.4% accuracy but 42.8% of LSTM-based and 30.1% of ZiPS. BERT is not only able to handle the numerical input with time series, but also performs stably and better than those traditional methods. Moreover, we found a new effective way in player performance prediction.

*Keywords— BERT, baseball, deep learning, long short-term memory, neural network, player performance prediction, projection system, Transformer*

### DATA

- BASEBALL REFERENCE: https://www.baseball-reference.com/
- Sean Lahman's database: http://www.seanlahman.com/baseball-archive/statistics/

### REQUIREMENTS

- pybaseball: https://github.com/jldbc/pybaseball
- Simple Transformers: https://github.com/ThilinaRajapakse/simpletransformers#text-classification
