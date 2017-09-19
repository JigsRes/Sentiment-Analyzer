#Read me
It implements Naive Bayes classifier to classify positive and negative sentiments. The training data is taken from IMDB reviews data. I have also implemented perceptron for classification.
 My code is tested for Python 2.7.12. It might give errors with Python 3.


  For Training
python NaiveBayes.py ../data/imdb1
python NaiveBayes.py -f ../data/imdb1
python NaiveBayes.py -b ../data/imdb1
python Perceptron.py ../data/imdb1/ 50

For Testing
python NaiveBayes.py ../data/imdb1 ../data/test_data
python Perceptron.py ../data/imdb1/ ../data/test_data 50 can be used to test perceptron
