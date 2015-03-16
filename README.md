1) Accuracy of my part-of-speech tagger is 95.759902
----------------------------------------------------

2) Precision, Recall and F-score for NER
----------------------------------------
(i) Precision, Recall and F score per class
-------------------------------------------
Class|Precision|Recall|F1-Score
--|--|--|--
PER|0.866372|0.801146|0.832483
LOC|0.636669|0.769309|0.696733
ORG|0.750329|0.671765|0.708876
MISC|0.462754|0.460674|0.461712

(ii) Overall f-score
--------------------
Overall F-score is 0.714071


3) What happens if we use Naive bayes instead of perceptron 
-----------------------------------------------------------

Part 1
-------
(i) POS tagging
---------------
Accuracy is 77.96943938978488

(ii) NER
--------
Class|Precision|Recall|F1-Score
--|--|--|--
LOC|0.229612|0.497371|0.314181
PER|0.263793|0.251645|0.257576
ORG|0.197504|0.540767|0.289334
MISC|0.037037|0.006803|0.011494

Overall F-score is 0.277339


Pat 2)Why do u think so
-----------------------
Naive bayes assumes features as bag of words. Which is not true. There is some relation between the words next to each other. Hence the performance degrades.
