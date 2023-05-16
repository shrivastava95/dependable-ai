B20AI015 B20AI013 B20AI024
Jaisidh Singh
Ishaan Shrivastava
Nakul Sharma
---------------------------------------------------------------------------------------------
Navigate to the main directory of the code folder provided in terminal, and run the following commands to reproduce the results of our approach.

python3 main.py \
       	--exp sample \
	--workers 2  \
	--epochs 50  \
	--start-epoch 0 \
	--batch-size 128 \
	--learning-rate 0.001 \
	--weight-decay 0 \
	--save-freq 1 \
	--print-iter 1 \
	--save-dir noise  \
	--test 0 \
	--test_e4 0 \
	--defense 1 \
	--optimizer adam

python3 eval.py \
       	--exp sample \
	--workers 2  \
	--epochs 10  \
	--start-epoch 0 \
	--batch-size 16 \
	--learning-rate 0.0003 \
	--weight-decay 0 \
	--save-freq 1 \
	--print-iter 1 \
	--save-dir noise  \
	--test 0 \
	--test_e4 0 \
	--defense 1 \
	--optimizer adam