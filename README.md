# Using BERT on Prediction Problems with Numeric Input Data: the Case of Major League Baseball Home Run Prediction
This repository includes the necessary implementation to reproduce the experiments highlighted in the my master's thesis.

## Abstract
BERT is a powerful deep learning model in natural language processing. It performs well in various language tasks such as machine translation and question answering since it has a great ability to analyze word sequences. In this paper, we show that BERT is able to make a prediction with numerical data input instead of text. We want to predict output with numerical data and verify its performance. In particular, we choose the home run performance prediction task which inputs the stats of players in Major League Baseball. We also compare the result of the BERT-based approach with the performance of the LSTM-based model and the popular projection system ZiPS. In testing data of the year 2018, the Bert-based approach reaches 50.6% accuracy while the LSTM-based model has 48.8% and ZiPS gets only 25.4% accuracy rate. In 2019, BERT achieves 44.4% accuracy but 42.8% of LSTM-based and 30.1% of ZiPS. BERT is not only able to handle the numerical input with time series, but also performs stably and better than those traditional methods. Moreover, we found a new effective way in player performance prediction.

***Keywords— BERT, baseball, deep learning, long short-term memory, neural network, player performance prediction, projection system, Transformer***
 
## Getting Started 

### Prerequisites

- Python 3.6
- Keras
- numpy
- pandas
- h5py

### Install
- Simple Transformers: https://github.com/ThilinaRajapakse/simpletransformers#text-classification

#### optional
- pybaseball: https://github.com/jldbc/pybaseball

## DATA
We collected data from the following two websites:
- BASEBALL REFERENCE: https://www.baseball-reference.com/
- Sean Lahman's database: http://www.seanlahman.com/baseball-archive/statistics/


```pybaseball``` is also an useful module to get players' data easily.

We collect players recorded in MLB with 20 features for every year during the period from 1998 to 2019. Since the final two teams Arizona Diamondbacks and the American Tampa Bay Devil Rays added in 1998, we started to collect data from that year. All players have at least 6 continuous years’ experience in MLB which means they have at least 1 Plate Appearances (PA) for 6 continued seasons. These features include 17 annual accumulated performance and 3 player’s status. 

Since home runs (HR) is the most effective way to score in the ball game, we set it as the main goal to predict in this paper. However, it is not only practically impossible but inefficient to predict the precision number of HRs. Therefore, we spilt HRs into disjoint subsets with every 5 HRs and each of them will be a class. We choose these classes to be our outputs. For our data, we have 12 classes. Following table shows our classes:
|Class| Numbers | Class | Numbers|
|---|---|---|---|
|$$C_1$$ | 0-4 | $$C_7$$ | 30-34|
|$$C_2$$ | 5-9 | $$C_8$$ | 35-39|
|$$C_3$$ | 10-14 | $$C_9$$ | 40-44|
|$$C_4$$ | 15-19 | $$C_{10}$$ | 45-49|
|$$C_5$$ | 20-24 | $$C_{11}$$ | 50-54|
|$$C_6$$ | 25-29 | $$C_{12}$$ | 55-59|


On the other side, we think 5-years-long data is stable enough to be input for models to know the player well. Therefore, we use 5 continued years stat with 20 features for each year of a player to predict his HRs class in the sixth year.

20 features include: Player's age, Player's Height, Player's Weight, Games Played, Plate Appearances, Runs Scored, Hits, Double Hits, Triple Hits, Home Runs, Runs Battle In, Stolen Bases, Caught Stealing, Bases on Balls, StrikeOuts, Grouned into Double Plays, Hit By a Pitch, Sacrifice Hits, Sacrifice Flies, Intended Bases on Balls,


## Models
### LSTM
|| LSTM | D/BN | TD | FC | D/BN | FC |
|---|---|---|---|---|---|---|
|A | 128\*2 | 0.25 | 10 | 1024 | 0.25 | 128 |
|B | 128\*2 | 0.25 | | 1024 | 0.25 | 128 |
|C | 128 | 0.25 | 10 | 1024 | 0.25 | 128 |
|D | 128\*2 | 0.25 | | 1024 | 0.25 | |
|E | 128\*2 | $$\surd$$ | 10 | 1024 | $$\surd$$ | 128 |
|F | 32 | | | 1024 | | |
|G | 64\*2 | 0.25 | | 1024 | 0.25 | 128 |
|H | 128\*2 | 0.25 | | 512 | 0.25 | 64 |
|I | 128 | | | | | |
|J | 128 | | | 1024 | $$\surd$$ | |


Each model contains the LSTM layer and fully connected neural networks with BN or dropout(D). In LSTM column, $$n*k$$ means $$k$$ layers with $$n$$ neurons. In D/BN columns, number represents dropout rate and $\surd$ means BN is used instead of dropout. In model A, C and E, there is a timestep-wise dimension reduction (TD) in the middle which reduced the dimension. Finally, a standard fully connected layer with softmax will be our output. In all fully connected layers we use ReLU as the activation function to boost the training speed. The loss is counted by cross-entropy and optimizer is Adam with learning rate $$10^{-8}$$. We set the batch size for 14 and trained for 20000 epochs.


### BERT
There are two kinds of pre-trained models of BERT we use: BERT-base and BERT-base-multilingual. Each of them has the version of cased and uncased which the wordpiece has the capital sub-words or not. We will fine-tune these pre-trained models by our task for 10 epochs. BERT-base has 12 layers, 768 hidden neurons and 12 attention heads for total 110 millions of parameters. It trained on English texts. BERT-base-multilingual-cased (BERT-mc) was trained on 104 languages based on Bert-cased and BERT-base-multilingual-uncased (BERT-mu) was trained on 102 languages based on Bert-cased. We set batch size for 14 with learning rate for $$10^{-5}$$. 

### sZymborski Projection System (ZiPS)
sZymborski Projection System (ZiPS) created by Dan Szymborski weights heavily for recent years which included velocity and pitch data for multiple years. Both are similar to Marcel. Furthermore, ZiPS also use comparable players to adjust the prediction. You can find more information in [here](https://www.fangraphs.com/projections.aspx?pos=all&stats=bat&type=zipsp1&team=0&lg=all&players=0).

## Training
We use 892 data points from 1998 to 2017 as training data and 265 data points in 2018 and 268 datas in 2019 for testing which verify our model performance. The input of testing data points in 2018 is in the period of year 2013 to 2017 and output HR class is belonging to year 2018.Testing data points in 2019 are similar. We also retrain our model after adding 265 data points from 2018 into training data and predict data in 2019. ZiPS has 209 predictions in 2018 and 215 in 2019. 


## Result
### Model Performance
We use following four rates to evaluate our models performance:
$$\begin{align*}
{\rm MPA} &= \frac{{\rm Right~Prediction}}{{\rm Total~Data~Points}}\\
{\rm H1C} &= \frac{{\rm Higher~Prediction~For~1~Class}}{{\rm Total~Data~Points}}\\
{\rm L1C} &= \frac{{\rm Lower~Prediction~For~1~Class}}{{\rm Total~Data~Points}}\\
{\rm TPA} &= {\rm MPA} + {\rm H1C} + {\rm L1C}
\end{align*}$$

#### Prediction accuracy of 2018
|| MPA(\%) | H1C(\%) | L1C(\%) | TPA(\%) |
|---|---|---|---|---|
|LSTM B | 45.7 | 12.8 | 12.5 | 71.0 |
|LSTM F | 48.7 | 13.6 | 12.5 | 74.8 |
|ZiPS | 25.4 | 11.0 | 37.3 | 73.7 |
|BERT-c | 46.8 | 12.8 | 13.6 | 73.2 |
|BERT-u | 49.1 | 9.4 | 13.6 | 72.1 |
|BERT-mc | $$\bf {50.6}$$ | 16.2 | 12.5 | $${\bf 79.3}$$ |
|BERT-mu | 47.6 | 16.6 | 12.1 | 76.3 |

#### Prediction accuracy of 2019 after retrain
|| MPA(\%) | H1C(\%) | L1C(\%) | TPA(\%) |
|---|---|---|---|---|
|LSTM A | 42.2 | 7.5 | 17.2 | 66.9 |
|LSTM D | 42.8 | 7.8 | 13.8 | 63.4 |
|ZiPS | 30.1 | 16.0 | 19.7 | 65.8 |
|BERT-c | 39.6 | 12.3 | 15.7 | 67.6 |
|BERT-u | 39.2 | 13.4 | 17.9 | $${\bf 70.5}$$ |
|BERT-mc | $${\bf 44.4}$$ | 9.7 | 15.3 | 69.4 |
|BERT-mu | 42.5 | 10.1 | 15.3 | 67.9 |


### Class Result
#### Number of correct predictions in 2018
|| 0-4 | 5-9 | 10-14 | 15-19 | 20-24 | 25-29 | 30+ |
|---|---|---|---|---|---|---|---|
|LSTM B | 79 | 18 | 15 | 0 | 9 | 0 | 0 |
|LSTM F | 79 | 22 | 16 | 0 | 11 | 1 | 0 |
|BERT-c | 83 | $${\bf 22}$$ | 8 | 0 | 11 | 0 | 0  |
|BERT-u | $${\bf 89}$$ | 17 | 14 | 1 | 8 | 1 | 0  |
|BERT-mc | 84 | 17 | $${\bf 23}$$ | 0 | 9 | 1 | 0  |
|BERT-mu | 81 | 17 | 11 | 0 | $${\bf 17}$$ | 0 | 0  |
|Ground Truth | 107 | 43 | 41 | 25 | 27 | 8 |14 |

#### Number of correct predictions in 2019 after retrain
|| 0-4 | 5-9 | 10-14 | 15-19 | 20-24 | 25-29 | 30+ |
|---|---|---|---|---|---|---|---|
|LSTM A | $${\bf 81}$$ | 18 | 4 | 0 | 9 | 1 | 0 |
|LSTM D | 80 | 14 | 11 | 0 | 7 | 0 | 0 |
|BERT-c | 74 | 19 | 8 | 1 | 3 | 1 | 0  |
|BERT-u | 71 | $${\bf 22}$$ | 4 | 0 | 7 | 1 | 0  |
|BERT-mc | 75 | 18 | $${\bf 12}$$ | 0 | $${\bf 13}$$ | 1 | 0  |
|BERT-mu | 73 | 22 | $${\bf 12}$$ | 0 | 6 | 1 | 0  |
|Ground Truth | 102 | 35 | 39 | 24 | 28 | 13 | 27 |

#### Number of correct predictions of ZiPS
|| 0-4 | 5-9 | 10-14 | 15-19 | 20-24 | 25-29 | 30+ |
|---|---|---|---|---|---|---|---|
|ZiPS 2018 | 14 | 7 | 12 | 7 | 6 | 2 | 5  |
| Ground Truth 2018| 107 | 43 | 41 | 25 | 27 | 8 |14 |
|ZiPS 2019 | 15 | 17 | 14 | 6 | 7 | 2 | 3 |
| Ground Truth 2019| 102 | 35 | 39 | 24 | 28 | 13 | 27 |


## Conclusion
- In this paper, we research the possibility of using numerical data with order to be the input of the powerful NLP model BERT. We predict the class of HRs number of MLB players by their past performance.
- Performance of BERT and LSTM are amazing. All models have outperformed ZiPS in the maximum prediction accuracy.
- Moreover, models based on BERT are usually have better performance than those based on LSTM.
- In conclusion, BERT is a feasible and effective way to predict output by numerical input.

## Future Work
- The structure of BERT can be adjusted due to numerical data input to increase accuracy and be better than LSTM.
- Creating an embedding for baseball player to know more information about them. 
- Solving the prediction result of imbalanced data.

## Authors

- **Husan-Cheng Sun** sun01120112@gmail.com
- **Advisor: Dr. Yen-Lung Tsai** yenlung@nccu.edu.tw