	_??,?-@_??,?-@!_??,?-@	?I?^????I?^???!?I?^???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:_??,?-@?n?o?>@A%@7???Y?X?vM??rEagerKernelExecute 0*	??Q??N@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?f??j+??!?ކ?t?A@)4??s??1l6yJ?J@@:Preprocessing2F
Iterator::Model*??g\??!???<ahC@)S?!?uq??1^t?t??5@:Preprocessing2U
Iterator::Model::ParallelMapV2???1ZG??!????0@)???1ZG??1????0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceF%u?{?!J??u?%@)F%u?{?1J??u?%@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?*??p???!
?f
3@)j0?G?t?1?nW? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip333333??!C:Þ?N@)???4q?1MY}?G@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?y?Cn?[?!??ڐ??@)?y?Cn?[?1??ڐ??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 86.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?I?^???I???b?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?n?o?>@?n?o?>@!?n?o?>@      ??!       "      ??!       *      ??!       2	%@7???%@7???!%@7???:      ??!       B      ??!       J	?X?vM???X?vM??!?X?vM??R      ??!       Z	?X?vM???X?vM??!?X?vM??b      ??!       JCPU_ONLYY?I?^???b q???b?X@