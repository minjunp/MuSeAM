Procedure for analysis (using MaxPool output)
Step 1-a: Get maxpool outputs from main.py(pooling_layer function)
Step 1-b: Get coordinate information for synthetic-removed sequences (step1.py)

Step 2-a: Get maxpool hitting coordinates from main.py(pooling_coordinate)
Step 2-b: Transform pooling coordinates into 512 filters (step2.py)

Step 3: Combine above information (unnecessary)
Step 3: Calculate binom test (step3.py)

Step 4: Identify significant pair and match with tomtom (step4.py)

Step 5: Compare with bioGRID (step5.py)

Procedure for analysis (using ReLU output)
Step 1-a: Get ReLU outputs from main.py(relu_layer function)
Step 1-b: Get coordinate information for synthetic-removed sequences (step1.py)

(No step 2)

step 3:
