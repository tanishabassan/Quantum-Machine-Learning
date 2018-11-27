from datasets import *
from qiskit_aqua.utils import split_dataset_to_data_and_labels
from qiskit_aqua.input import get_input_instance
from qiskit_aqua import run_algorithm
import numpy as np

n = 2  # dimension of each data point
sample_Total, training_input, test_input, class_labels = Breast_cancer(training_size=40,
                                                              test_size=10, n=n, PLOT_DATA=True)

temp = [test_input[k] for k in test_input]
total_array = np.concatenate(temp)

aqua_dict = {
    'problem': {'name': 'svm_classification', 'random_seed': 100},
    'algorithm': {
        'name': 'QSVM.Kernel'
    },
    'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2, 'entangler_map': {0: [1]}},
    'multiclass_extension': {'name': 'AllPairs'},
    'backend': {'name': 'qasm_simulator', 'shots': 256}
}

algo_input = get_input_instance('SVMInput')
algo_input.training_dataset = training_input
algo_input.test_dataset = test_input
algo_input.datapoints = total_array

result = run_algorithm(aqua_dict, algo_input)
for k,v in result.items():
    print("'{}' : {}".format(k, v))
