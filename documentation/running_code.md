
# Train RBM
+ modify the call to training algo based on ypur requirement
```py
train_rbm(num_hid=4000, start_epoch=0, end_epoch=3600, skip_iter=50)
```
+ then run the code
```py
python rbm_learn_model.py
```
+ this will dump learned weight parameters after every 50 iterations inside
`WriterIdentification\data\new_representation` this will go on till iteration 3600
+ If training crashes due less disk space in some intermediate epoch you can easily resume by modifying `start_epoch`
value

# Run benchmark

+ plug any classifier you want in method `get_classifier_dict()` we have initialized it for you already
+ then run the code
```py
python benchmark.py
```
+ the code will run and dump the classification performance results in Report folder
+ Note that the default configuration will run entire code as mentioned in our framework and will take lot of time.
You can selectively switch off certain parts of code.

# Analyzing the RBM learning
+ running the code will dump some csv files in report folder
+ you need to visualize these file in some external tools
+ I have used plot.ly (https://plot.ly)

# Running the unittests
+ for some critical functionalities unittestts are implemented
+ Run it by
```py
python unit_tests.py
```