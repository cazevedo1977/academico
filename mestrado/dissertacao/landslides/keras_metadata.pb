
�troot"_tf_keras_network*�t{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "slope"}, "name": "slope", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "aspect"}, "name": "aspect", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "elevation"}, "name": "elevation", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "twi"}, "name": "twi", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "curv"}, "name": "curv", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lito"}, "name": "lito", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "uso_solo"}, "name": "uso_solo", "inbound_nodes": []}, {"class_name": "Normalization", "config": {"name": "normalization_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_1", "inbound_nodes": [[["slope", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_2", "inbound_nodes": [[["aspect", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_3", "inbound_nodes": [[["elevation", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_4", "inbound_nodes": [[["twi", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_5", "inbound_nodes": [[["curv", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_6", "inbound_nodes": [[["lito", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_7", "inbound_nodes": [[["uso_solo", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["normalization_1", 0, 0, {}], ["normalization_2", 0, 0, {}], ["normalization_3", 0, 0, {}], ["normalization_4", 0, 0, {}], ["normalization_5", 0, 0, {}], ["normalization_6", 0, 0, {}], ["normalization_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 12, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4349508873549036, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["slope", 0, 0], ["aspect", 0, 0], ["elevation", 0, 0], ["twi", 0, 0], ["curv", 0, 0], ["lito", 0, 0], ["uso_solo", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "shared_object_id": 22, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "slope"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "aspect"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "elevation"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "twi"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "curv"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "lito"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "uso_solo"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "slope"}, "name": "slope", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "aspect"}, "name": "aspect", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "elevation"}, "name": "elevation", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "twi"}, "name": "twi", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "curv"}, "name": "curv", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lito"}, "name": "lito", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "uso_solo"}, "name": "uso_solo", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "Normalization", "config": {"name": "normalization_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_1", "inbound_nodes": [[["slope", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Normalization", "config": {"name": "normalization_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_2", "inbound_nodes": [[["aspect", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Normalization", "config": {"name": "normalization_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_3", "inbound_nodes": [[["elevation", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Normalization", "config": {"name": "normalization_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_4", "inbound_nodes": [[["twi", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Normalization", "config": {"name": "normalization_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_5", "inbound_nodes": [[["curv", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Normalization", "config": {"name": "normalization_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_6", "inbound_nodes": [[["lito", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Normalization", "config": {"name": "normalization_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "name": "normalization_7", "inbound_nodes": [[["uso_solo", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["normalization_1", 0, 0, {}], ["normalization_2", 0, 0, {}], ["normalization_3", 0, 0, {}], ["normalization_4", 0, 0, {}], ["normalization_5", 0, 0, {}], ["normalization_6", 0, 0, {}], ["normalization_7", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 12, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4349508873549036, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 21}], "input_layers": [["slope", 0, 0], ["aspect", 0, 0], ["elevation", 0, 0], ["twi", 0, 0], ["curv", 0, 0], ["lito", 0, 0], ["uso_solo", 0, 0]], "output_layers": [["dense_1", 0, 0]]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": true, "label_smoothing": 0}, "shared_object_id": 30}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 31}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.0013054978335276246, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "slope", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "slope"}}2
�root.layer-1"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "aspect", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "aspect"}}2
�root.layer-2"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "elevation", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "elevation"}}2
�root.layer-3"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "twi", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "twi"}}2
�root.layer-4"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "curv", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "curv"}}2
�root.layer-5"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "lito", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lito"}}2
�root.layer-6"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "uso_solo", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "uso_solo"}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "inbound_nodes": [[["slope", 0, 0, {}]]], "shared_object_id": 7, "build_input_shape": [null, 1]}2
�	root.layer_with_weights-1"_tf_keras_layer*�{"name": "normalization_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "inbound_nodes": [[["aspect", 0, 0, {}]]], "shared_object_id": 8, "build_input_shape": [null, 1]}2
�
root.layer_with_weights-2"_tf_keras_layer*�{"name": "normalization_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "inbound_nodes": [[["elevation", 0, 0, {}]]], "shared_object_id": 9, "build_input_shape": [null, 1]}2
�root.layer_with_weights-3"_tf_keras_layer*�{"name": "normalization_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "inbound_nodes": [[["twi", 0, 0, {}]]], "shared_object_id": 10, "build_input_shape": [null, 1]}2
�root.layer_with_weights-4"_tf_keras_layer*�{"name": "normalization_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "inbound_nodes": [[["curv", 0, 0, {}]]], "shared_object_id": 11, "build_input_shape": [null, 1]}2
�
�root.layer_with_weights-6"_tf_keras_layer*�{"name": "normalization_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}}, "inbound_nodes": [[["uso_solo", 0, 0, {}]]], "shared_object_id": 13, "build_input_shape": [null, 1]}2
�
�root.layer_with_weights-7"_tf_keras_layer*�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 12, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7]}}2
�
�root.layer_with_weights-8"_tf_keras_layer*�{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}}2
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 34}2
��root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 31}2