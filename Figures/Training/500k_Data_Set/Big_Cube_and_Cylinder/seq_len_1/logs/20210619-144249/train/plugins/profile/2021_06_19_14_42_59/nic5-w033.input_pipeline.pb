	?????V%@?????V%@!?????V%@	e?L@???e?L@???!e?L@???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?????V%@?@?ش?#@A?,σ????Y%y???A??rEagerKernelExecute 0*	`??"?QU@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?e??ۡ?!/?T?.sD@)0???hȠ?1Ţ??7C@:Preprocessing2F
Iterator::Model??E_A???!Nf I(D@)0L?
F%??1 ??m?68@:Preprocessing2U
Iterator::Model::ParallelMapV2t??gy??!?#ӟ0@)t??gy??1?#ӟ0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceE?a??x?!???c?@)E?a??x?1???c?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapиp $??!q?M?|?+@)Zd;?O?w?12??je?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip\W?o??!ⱙ߶?M@)?n???q?1????E@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4a?!??6?@)?J?4a?1??6?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 92.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9e?L@???Iz????X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?@?ش?#@?@?ش?#@!?@?ش?#@      ??!       "      ??!       *      ??!       2	?,σ?????,σ????!?,σ????:      ??!       B      ??!       J	%y???A??%y???A??!%y???A??R      ??!       Z	%y???A??%y???A??!%y???A??b      ??!       JCPU_ONLYYe?L@???b qz????X@