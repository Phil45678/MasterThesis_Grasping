?	vS?k%?@vS?k%?@!vS?k%?@	ߘ???@ߘ???@!ߘ???@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:vS?k%?@t?Lh???Al[?? ???Y??'?b??rEagerKernelExecute 0*aX9??N@)       =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?;? є?!???@@)?Z?kBZ??1?S?I??>@:Preprocessing2F
Iterator::Model??E;???!8??^mD@) ?d?F ??1????8@:Preprocessing2U
Iterator::Model::ParallelMapV2Ac&Q/???!?(?<?0@)Ac&Q/???1?(?<?0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicea??+ey?!????#N$@)a??+ey?1????#N$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???Mb??!????=3@)A??ǘ?v?1&??k,-"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip^H??0~??!??o$??M@)?p>??p?1d*`?Yv@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor=???mW?!D???@)=???mW?1D???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 72.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ߘ???@Ir?#r?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	t?Lh???t?Lh???!t?Lh???      ??!       "      ??!       *      ??!       2	l[?? ???l[?? ???!l[?? ???:      ??!       B      ??!       J	??'?b????'?b??!??'?b??R      ??!       Z	??'?b????'?b??!??'?b??b      ??!       JCPU_ONLYYߘ???@b qr?#r?W@Y      Y@q???@"?
both?Your program is POTENTIALLY input-bound because 72.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 