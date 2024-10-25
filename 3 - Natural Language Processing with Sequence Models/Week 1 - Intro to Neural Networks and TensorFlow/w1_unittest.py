import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
import traceback

import numpy as np
from test_utils import comparator, summary

def test_build_vocabulary(target):

    test_cases = [
        {
            "name": "simple_test_check1",
            "input": {
                "corpus": [['a']],
            },
            "expected": {
                "output_list": {'': 0, '[UNK]': 1, 'a': 2},
                "output_type": type(dict()),
            },
        },
        {
            "name": "simple_test_check2",
            "input": {
                "corpus": [['a', 'aa'], ['a', 'ab'], ['ccc']],
            },
            "expected": {
                "output_list": {'': 0, '[UNK]': 1, 'a': 2, 'aa': 3, 'ab': 4, 'ccc': 5},
                "output_type": type(dict()),
            },
        },
    ]

    failed_cases = []
    successful_cases = 0

    for test_case in test_cases:

        result = target(**test_case["input"])

        try:
            assert result == test_case["expected"]["output_list"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output does not match with expected values. Maybe you can check the value you are using for unk_token variable. Also, try to avoid using the global dictionary Vocab.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:

            assert isinstance(result, test_case["expected"]["output_type"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output object does not have the correct type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_max_length(target):

    test_cases = [
        {
            "name": "simple_test_check1",
            "input": {
                "training_x": [['cccc']],
                "validation_x": [['a', 'aa'], ['a', 'ab'], ['cccc']],
            },
            "expected": {
                "output_list": 2,
                "output_type": type(1),
            },
        },
        {
            "name": "simple_test_check2",
            "input": {
                "training_x": [['a', 'aa'], ['a', 'ab'], ['ccc'], ['ddd']],
                "validation_x": [['a', 'aa'], ['a', 'ab', 'ac'], ['ccc']],
            },
            "expected": {
                "output_list": 3,
                "output_type": type(1),
            },
        },
    ]

    failed_cases = []
    successful_cases = 0

    for test_case in test_cases:

        result = target(**test_case["input"])

        try:
            assert result == test_case["expected"]["output_list"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output does not match with expected values. Make sure you are measuring the length of tweets and not the length of datasets or words. Expected: {failed_cases[-1].get('expected')}.\n Got: {failed_cases[-1].get('got')}."
            )

        try:

            assert isinstance(result, test_case["expected"]["output_type"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output object does not have the correct type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_padded_sequence(target):

    test_cases = [
        {
            "name": "simple_test_check1",
            "input": {
                "tweet": ['a', 'a', 'aaa', 'cats'],
                "vocab_dict": {
                     '': 0,
                     '[UNK]': 1,
                     'a': 2,
                     'aa': 3,
                     'aaa': 4,
                     'aaaa': 5,
                },
                "max_len": 5,
                "unk_token": '[UNK]'
            },
            "expected": {
                "output_list": [2, 2, 4, 1, 0],
                "output_type": type([]),
            },
        },
    ]

    failed_cases = []
    successful_cases = 0

    for test_case in test_cases:

        result = target(**test_case["input"])

        try:
            assert result == test_case["expected"]["output_list"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output does not match with expected values. Maybe you can check the value you are using for unk_token variable. Also, try to avoid using the global dictionary Vocab.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(result, test_case["expected"]["output_type"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output object does not have the correct type.\n Expected: {failed_cases[-1].get('expected')}.\n Got: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert len(result) == len(test_case["expected"]["output_list"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output object does not have the correct length.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")        


def test_relu(target):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "check_output1",
            "input": np.array([[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0]], dtype=float),
            "expected": {
                "values": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]]),
                "shape": (2, 3),
            },
        },
        {
            "name": "check_output2",
            "input": np.array(
                [
                    [-3.0, 1.0, -5.0, 4.0],
                    [-100.0, 3.0, -2.0, 0.0],
                    [-4.0, 0.0, 1.0, 5.0],
                ],
                dtype=float,
            ),
            "expected": {
                "values": np.array(
                    [[0.0, 1.0, 0.0, 4.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 1.0, 5.0]]
                ),
                "shape": (3, 4),
            },
        },
    ]

    relu_layer = target

    for test_case in test_cases:
        result = relu_layer(test_case["input"])
        try:
            assert result.shape == test_case["expected"]["shape"]
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["shape"],
                    "got": result.shape,
                }
            )
            print(
                f"Relu should not modify the input shape.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"]["values"],)
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["values"],
                    "got": result,
                }
            )
            print(
                f"Output from relu is incorrect.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_sigmoid(target):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "check_output1",
            "input": np.array([[-1000.0, -1.0, 0.0], [0.0, 1.0, 1000.0]], dtype=float),
            "expected": {
                "values": np.array([[0.0, 0.26894142, 0.5], [0.5, 0.73105858, 1.0]]),
                "shape": (2, 3),
            },
        },
        {
            "name": "check_output2",
            "input": np.array(
                [
                    [-3.0, 1.0, -5.0, 4.0],
                    [-100.0, 3.0, -2.0, 0.0],
                    [-4.0, 0.0, 1.0, 5.0],
                ],
                dtype=float,
            ),
            "expected": {
                "values": np.array(
                    [[4.74258732e-02, 7.31058579e-01, 6.69285092e-03, 9.82013790e-01],
                     [3.72007598e-44, 9.52574127e-01, 1.19202922e-01, 5.00000000e-01],
                     [1.79862100e-02, 5.00000000e-01, 7.31058579e-01, 9.93307149e-01]]
                ),
                "shape": (3, 4),
            },
        },
    ]

    relu_layer = target

    for test_case in test_cases:
        result = relu_layer(test_case["input"])
        try:
            assert result.shape == test_case["expected"]["shape"]
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["shape"],
                    "got": result.shape,
                }
            )
            print(
                f"Sigmoid function should not modify the input shape.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"]["values"],)
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["values"],
                    "got": result,
                }
            )
            print(
                f"Output from sigmoid function is incorrect.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_Dense(target):
    failed_cases = []
    successful_cases = 0
    
    input1 = np.array([[2.0, 7.0]])
    input2 = np.array([[2.0, 7.0, -10], [-1, -2, -3]])
    
    test_cases = [
        {
            "name": "simple_test_check1",
            "input_init": {
                "n_units": 2, 
                "input_shape": input1.shape,
                "activation": lambda x: np.maximum(x, 0)
            },
            "input_forward": {
                "x": input1, 
            },
            "expected": {
                "weights": np.array(
                    [[ 0.03047171, -0.10399841],
                    [ 0.07504512,  0.09405647]]
                ),
                "output": np.array([[0.58625925, 0.45039848]]),
            },
        },
        {
            "name": "simple_test_check2",
            "input_init": {
                "n_units": 2, 
                "input_shape": input2.shape,
                "activation": lambda x: np.maximum(x, 0)
            },
            "input_forward": {
                "x": input2, 
            },
            "expected": {
                "weights": np.array(
                    [[ 0.03047171, -0.10399841],
                     [ 0.07504512,  0.09405647],
                     [-0.19510352, -0.13021795]]
                ),
                "output": np.array(
                    [[2.53729444, 1.75257799],
                     [0.40474861, 0.30653932]]),
            },
        },
        
    ]

    for test_case in test_cases:
        dense_layer = target(**test_case["input_init"])
        result = dense_layer(**test_case["input_forward"])
        
        try:
            assert dense_layer.weights.shape == test_case["expected"]["weights"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["weights"].shape,
                    "got": dense_layer.weights.shape,
                }
            )
            print(
                f"Weights matrix has the incorrect shape.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            
        try:
            assert np.allclose(dense_layer.weights, test_case["expected"]["weights"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["weights"],
                    "got": dense_layer.weights,
                }
            )
            print(
                f"The weights did not initialize correctly. Make sure you did not change the random seed.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"]["output"],)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["output"],
                    "got": result,
                }
            )
            print(
                f"Dense layer produced incorrect output. Check your weights or your output computation.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(result, type(test_case["expected"]["output"]))
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["output"]),
                    "got": type(result),
                }
            )
            print(
                f"Output object has the incorrect type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_model(target):
    successful_cases = 0
    failed_cases = []
    
    dummy_layers = [
        tf.keras.layers.Embedding(1, 2, input_length=3),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(2, activation='relu')
    ]
    dummy_model = tf.keras.Sequential(dummy_layers)
    dummy_model.compile()
    
    test_cases = [
        {
            "name": "simple_test_check1",
            "input": {"num_words": 20, "embedding_dim": 16, "max_len": 5},
            "expected": {
                "type": type(dummy_model),
                "no_layers": 3,
                "layer_1_type": type(dummy_layers[0]),
                "layer_1_input_dim": 20,
                "layer_1_input_length": 5,
                "layer_1_output_dim": 16,
                "layer_2_type": type(dummy_layers[1]),
                "layer_3_type": type(dummy_layers[2]),
                "layer_3_output_shape": (None, 1),
                "layer_3_activation": tf.keras.activations.sigmoid
            },
        },
        
    ]

    for test_case in test_cases:
        
        model = target(**test_case["input"])

        try:
            assert isinstance(model, test_case["expected"]["type"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["type"],
                    "got": type(model),
                }
            )

            print(
                f"Wrong type of the returned model.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}"
            )

        try:
            assert len(model.layers) == test_case["expected"]["no_layers"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["no_layers"],
                    "got": len(model.layers),
                }
            )
            print(
                f"The model has an incorrect number of layers.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected:{failed_cases[-1].get('expected')}"
            )

        try:
            assert isinstance(model.layers[0], test_case["expected"]["layer_1_type"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_1_type"],
                    "got": type(model.layers[0]),
                }
            )
            print(
                f"The first layer has incorrect type.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}"
            )
            
        try:
            assert model.layers[0].input_dim == test_case["expected"]["layer_1_input_dim"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_1_input_dim"],
                    "got": model.layers[0].input_dim,
                }
            )
            print(
                f"The first layer has wrong input dimensions.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}"
            ) 
            
        try:
            assert model.layers[0].input_length == test_case["expected"]["layer_1_input_length"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_1_input_length"],
                    "got": model.layers[0].input_length,
                }
            )
            print(
                f"The first layer has wrong input length.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}"
            )      
            
        try:
            assert model.layers[0].output_dim == test_case["expected"]["layer_1_output_dim"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_1_output_dim"],
                    "got": model.layers[0].output_dim,
                }
            )
            print(
                f"The first layer has wrong output (embedding) dimension.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}"
            )

        try:
            assert isinstance(model.layers[1], test_case["expected"]["layer_2_type"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_2_type"],
                    "got": type(model.layers[1]),
                }
            )
            print(
                f"The second layer has incorrect type.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}",
                test_case["expected"]["expected_type"],
            )
                              
        try:
            assert isinstance(model.layers[2], test_case["expected"]["layer_3_type"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_3_type"],
                    "got": type(model.layers[2]),
                }
            )
            print(
                f"The third layer has incorrect type.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}"
            )

        try:
            assert model.layers[2].output_shape == test_case["expected"]["layer_3_output_shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_3_output_shape"],
                    "got": model.layers[2].output_shape,
                }
            )
            print(
                f"The last layer has wrong output shape.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected:{failed_cases[-1].get('expected')}"
            )
            
        try:
            assert model.layers[2].activation == test_case["expected"]["layer_3_activation"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["layer_3_activation"],
                    "got": model.layers[2].activation,
                }
            )
            print(
                f"The last layer has wrong output shape.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected: {failed_cases[-1].get('expected')}"
            )
            
    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_line_to_tensor(target):
    line = '10101110'
    vocab = sorted(set(line))
    vocab.insert(0, "[UNK]") # Add a special character for any unknown
    vocab.insert(1,"") # Add the empty character for padding.
    
    ids = target(line, vocab)
    assert tf.is_tensor(ids), f"Wrong type, your function must return a Tensor"
    assert (ids.dtype == tf.int64), f"Wrong number type. Expected: {tf.int64} but got {ids.dtype}"
    assert len(ids) == len(line), f"Wrong length. Expected: {len(line)} but got {len(ids)}"
    assert tf.math.reduce_all(tf.equal(ids, [3, 2, 3, 2, 3, 3, 3, 2])), f"Unit test 1 failed. "
    
    line = "123"
    ids = target(line, vocab)
    assert tf.is_tensor(ids), f"Wrong type, your function must return a Tensor"
    assert len(ids) == len(line), f"Wrong length. Expected: {len(line)} but got {len(ids)}"
    assert tf.math.reduce_all(tf.equal(ids, [3, 0, 0])), f"Unit test 2 failed. "
    
    
    line = "123abc"
    vocab = sorted(set(line))
    vocab.insert(0, "[UNK]") # Add a special character for any unknown
    vocab.insert(1,"") # Add the empty character for padding.
    
    ids = target(line, vocab)
    assert tf.is_tensor(ids), f"Wrong type, your function must return a Tensor"
    assert len(ids) == len(line), f"Wrong length. Expected: {len(line)} but got {len(ids)}"
    assert tf.math.reduce_all(tf.equal(ids, [2, 3, 4, 5, 6, 7])), f"Unit test 1 failed. "
    
    line = "1234567"
    ids = target(line, vocab)
    assert tf.is_tensor(ids), f"Wrong type, your function must return a Tensor"
    assert len(ids) == len(line), f"Wrong length. Expected: {len(line)} but got {len(ids)}"
    assert tf.math.reduce_all(tf.equal(ids, [2, 3, 4, 0, 0, 0, 0])), f"Unit test 2 failed. "
    
    print("\033[92mAll test passed!")

def test_create_batch_dataset(target):
    BATCH_SIZE = 2
    SEQ_LENGTH = 8 
    lines = ['abc 123 xyz', 'Hello world!', '1011101']
    vocab = sorted(set('abcdefghijklmnopqrstuvwxyz12345'))
    
    tf.random.set_seed(272)
    dataset = target(lines, vocab, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)
    exp_shape = (BATCH_SIZE, SEQ_LENGTH)
    outputs = dataset.take(1)
    assert len(outputs) > 0, f"Wrong length. First batch must have 1 element. Got {len(outputs)}"
    for in_line, out_line in dataset.take(1):
        assert tf.is_tensor(in_line), "Wrong type. in_line extected to be a Tensor"
        assert tf.is_tensor(out_line), "Wrong type. out_line extected to be a Tensor"
        assert in_line.shape == exp_shape, f"Wrong shape in in_line. Expected {in_line.shape} but got: {exp_shape}"
        assert out_line.shape == exp_shape, f"Wrong shape. Expected {in_line.shape} but got: {exp_shape}"

        expected_in_line = [[28, 20, 23, 17,  9,  0,  0,  1],
                            [30, 31,  0,  0, 10, 17, 17, 20]]
        expected_out_line = [[20, 23, 17,  9,  0,  0,  1,  0],
                             [31,  0,  0, 10, 17, 17, 20,  0]]
        
        assert tf.math.reduce_all(tf.equal(in_line, expected_in_line)), \
            f"Wrong values. Expected {expected_in_line} but got: {in_line.numpy()}"
        assert tf.math.reduce_all(tf.equal(out_line, expected_out_line)), \
            f"Wrong values. Expected {expected_out_line} but got: {out_line.numpy()}"
        
    BATCH_SIZE = 4
    SEQ_LENGTH = 8 
    lines = [ 'Hello world!', '1918', '1010101', 'deeplearning.ai']
    vocab = sorted(set('abcdefghijklmnopqrstuvwxyz012345'))
    vocab.insert(0, "[UNK]") # Add a special character for any unknown
    vocab.insert(1,"") # Add the empty character for padding.
        
    tf.random.set_seed(5)
    dataset = target(lines, vocab, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)
    exp_shape = (BATCH_SIZE, SEQ_LENGTH)
    outputs = dataset.take(1)
    assert len(outputs) > 0, f"Wrong length. First batch must have 1 element. Got {len(outputs)}"
    for in_line, out_line in dataset.take(1):
        assert tf.is_tensor(in_line), "Wrong type. in_line extected to be a Tensor"
        assert tf.is_tensor(out_line), "Wrong type. out_line extected to be a Tensor"
        assert in_line.shape == exp_shape, f"Wrong shape in in_line. Expected {in_line.shape} but got: {exp_shape}"
        assert out_line.shape == exp_shape, f"Wrong shape. Expected {in_line.shape} but got: {exp_shape}"

        expected_in_line = [[19, 11,  0,  0,  3,  0,  3,  0],
                            [ 3,  2,  3,  2,  3,  2,  3,  0],
                            [ 0, 12, 19, 19, 22,  0, 30, 22],
                            [12, 12, 23, 19, 12,  8, 25, 21]]
        
        expected_out_line = [[11,  0,  0,  3,  0,  3,  0,  0],
                             [ 2,  3,  2,  3,  2,  3,  0, 11],
                             [12, 19, 19, 22,  0, 30, 22, 25],
                             [12, 23, 19, 12,  8, 25, 21, 16]]

        assert tf.math.reduce_all(tf.equal(in_line, expected_in_line)), \
            f"Wrong values. Expected {expected_in_line} but got: {in_line.numpy()}"
        assert tf.math.reduce_all(tf.equal(out_line, expected_out_line)), \
            f"Wrong values. Expected {expected_out_line} but got: {out_line.numpy()}"

    print("\n\033[92mAll test passed!")

def test_GRULM(target):
    batch_size = 64
    max_length = 128
    embedding_dim = 16
    vocab_size = 4
    rnn_units = 32
    modelw = target(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                rnn_units = rnn_units)
    print("Test case 1:")
    try:
        modelw.build(input_shape=(batch_size, max_length))
        modelw.call(Input(shape=(max_length)))
        comparator(summary(modelw),
                   [['Embedding', (None, max_length, embedding_dim), 64], 
                    ['GRU', [(None, max_length, rnn_units), (None, rnn_units)], 4800, 'return_sequences=True', 'return_state=True'], 
                    ['Dense', (None, max_length, vocab_size), 132, 'log_softmax']])
    except:
        print("\033[91m\nYour model is not building")
        
    batch_size = 32
    max_length = 50
    embedding_dim = 400
    vocab_size = 52
    rnn_units = 101
    modelw = target(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                rnn_units = rnn_units)
    print("Test case 2:")
    try:
        modelw.build(input_shape=(batch_size, max_length))
        modelw.call(Input(shape=(max_length)))
        comparator(summary(modelw),
                   [['Embedding', (None, max_length, embedding_dim), 20800], 
                    ['GRU', [(None, max_length, rnn_units), (None, rnn_units)], 152409, 'return_sequences=True', 'return_state=True'], 
                    ['Dense', (None, max_length, vocab_size), 5304, 'log_softmax']])

    except:
        print("\033[91m\nYour model is not building")
        traceback.print_exc()


def test_compile_model(target):
    
    model = target(tf.keras.Sequential())
    # Define the loss function. Use SparseCategoricalCrossentropy 
    loss = model.loss
    #loss = model.loss
    assert type(loss) == tf.losses.SparseCategoricalCrossentropy, f"Wrong loss function.  Expected {tf.losses.SparseCategoricalCrossentropy} but got {type(loss)}]"
    y_true = [1, 2]
    y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    loss_y = loss(y_true, y_pred)
    assert np.isclose(loss_y, 0.9868951), f"Wrong value for loss. Expected {0.9868951} but got {loss_y}. Check from_logits parameter."
    optimizer = model.optimizer
    assert type(optimizer) == tf.keras.optimizers.Adam, "Wrong optimizer"
    assert np.isclose(optimizer.learning_rate.numpy(), 0.00125), f"Wrong learning_rate. Expected {0.00125} but got {optimizer.learning_rate.numpy()}."
    
    print("\n\033[92mAll test passed!")


def test_log_perplexity(target):
    test_cases = [
        {
            "name": "example 1",
            "input": {
                "preds": tf.constant([[[0.1, 0.3, 0.7],
                                       [0.1, 0.3, 0.7],
                                       [0.1, 0.3, 0.7],
                                       [0.1, 0.3, 0.7],
                                       [0.1, 0.3, 0.7]]]),
                "target": tf.constant([[2, 2, 2, 2, 2]]),
            },
            "expected": -0.699999988079071,
        },
        {
            "name": "example 2",
            "input": {
                "preds": tf.constant([[[0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.]]]),
                "target": tf.constant([[1, 1, 1, 1, 1]]),
            },
            "expected": float("nan"),
        },
                {
            "name": "example 3",
            "input": {
                "preds": tf.constant([[[0.0, 1.0, 0.],
                                       [0.0, 1.0, 0.],
                                       [0.0, 1.0, 0.],
                                       [0.0, 1.0, 0.],
                                       [0.0, 1.0, 0.]]]),
                "target": tf.constant([[1, 1, 1, 1, 1]]),
            },
            "expected": float("nan"),
        },
        {
            "name": "example 4",
            "input": {
                "preds": tf.constant([[[0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.],
                                       [0.0, 0.0, 1.]]]),
                "target": tf.constant([[2, 2, 2, 2, 2]]),
            },
            "expected": -1.,
        },
        {
            "name": "example 5",
            "input": {
                "preds": tf.constant([[[1., 0., 0., 0., 0.],
                                       [0., 1., 0., 0., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 0., 0., 1., 0.],
                                       [0., 0., 1., 0., 1.],
                                       [0., 0., 0., 1., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 1., 0., 0., 0.]]]),
                "target": tf.constant([[0, 1, 2, 3, 4, 3, 2, 1]]),
            },
            "expected": -1.,
        },
        {
            "name": "example 6",
            "input": {
                "preds": tf.constant([[[1., 0., 0., 0., 0.],
                                       [0., 1., 0., 0., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 0., 0., 1., 0.],
                                       [0., 0., 1., 0., 1.],
                                       [0., 0., 0., 1., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 1., 0., 0., 0.]]]),
                "target": tf.constant([[0, 1, 2, 3, 4, 0, 1, 2]]),
            },
            "expected": -4./6.,
        },
        {
            "name": "example 7",
            "input": {
                "preds": tf.constant([[[1., 0., 0., 0., 0.],
                                       [0., 1., 0., 0., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 0., 0., 1., 0.],
                                       [0., 0., 1., 0., 1.],
                                       [0., 0., 0., 1., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 1., 0., 0., 0.]]]),
                "target": tf.constant([[1, 2, 3, 4, 0, 0, 0, 0]]),
            },
            "expected": 0,
        },
        {
            "name": "example 8, Batch of 1",
            "input": {
                "preds": tf.constant([[[0.1, 0.5, 0.4],
                                       [0.05, 0.9, 0.05],
                                       [0.2, 0.3, 0.5],
                                       [0.1, 0.2, 0.7],
                                       [0.2, 0.8, 0.1],
                                       [0.4, 0.4, 0.2],
                                       [0.5, 0.0, 0.5]]]),
                "target": tf.constant([[1, 2, 0, 2, 0, 2, 0]]),
            },
            "expected": -0.3083333329608043,
        },
        {
            "name": "Example 9. Batch of 2",
            "input": {
                "preds": tf.constant([[[0.1, 0.5, 0.4],
                                       [0.05, 0.9, 0.05],
                                       [0.2, 0.3, 0.5],
                                       [0.1, 0.2, 0.7],
                                       [0.2, 0.8, 0.1],
                                       [0.4, 0.4, 0.2],
                                       [0.5, 0.0, 0.5]],
                                     [[0.1, 0.5, 0.4],
                                       [0.2, 0.8, 0.1],
                                       [0.4, 0.4, 0.2],
                                       [0.5, 0.0, 0.5],
                                       [0.05, 0.9, 0.05],
                                       [0.2, 0.3, 0.5],
                                       [0.1, 0.2, 0.7]]]),
                "target": tf.constant([[1, 2, 0, 2, 0, 2, 0], [2, 1, 1, 2, 2, 0, 0]]),
            },
            "expected": -0.27916666759798925,
        }
    ]
    
    for testi in test_cases:
        test_in = testi["input"]
        expected = testi["expected"]
        output = target(test_in["preds"], test_in["target"])
        if np.isnan(expected):
            assert np.isnan(output), f"Fail in {testi['name']}. Expected {expected} but got {output}"
        else:
            assert np.allclose(output, expected), f"Fail in {testi['name']}. Expected {expected} but got {output}"

    print("\n\033[92mAll test passed!")


def test_GenerativeModel(target, model, vocab):
    tf.random.set_seed(272)
    gen = target(model, vocab, temperature=0.5)
    n_chars = 40
    pre = "SEFOE"
    text1 = gen.generate_n_chars(n_chars, pre)
    assert len(text1) == n_chars + len(pre) , f"Wrong length. Expected {n_chars + len(pre)} but got{len(text1)}"
    text2 = gen.generate_n_chars(n_chars, pre)
    assert len(text2) == n_chars + len(pre), f"Wrong length. Expected {n_chars + len(pre)} but got{len(text2)}"
    assert text1 != text2, f"Expected different strings since temperature is > 0.0"

    gen = target(model, vocab, temperature=0.0)
    n_chars = 40
    pre = "What is "
    text1 = gen.generate_n_chars(n_chars, pre)
    assert len(text1) == n_chars + len(pre) , f"Wrong length. Expected {n_chars + len(pre)} but got{len(text1)}"
    text2 = gen.generate_n_chars(n_chars, pre)
    assert len(text2) == n_chars + len(pre), f"Wrong length. Expected {n_chars + len(pre)} but got{len(text2)}"
    assert text1 == text2, f"Expected same strings since temperature is 0.0"
    
    n_chars = 100
    pre = "W"
    text_l = gen.generate_n_chars(n_chars, pre)
    used_voc = set(text_l)
    assert used_voc.issubset(set(vocab)), "Something went wrong. Only characters in vocab can be produced." \
    f" Unexpected characters: {used_voc.difference(set(vocab))}"
    
    print("\n\033[92mAll test passed!")

