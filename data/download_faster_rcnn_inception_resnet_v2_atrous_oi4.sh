wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz
tar -xvzf faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz
mv faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12/frozen_inference_graph.pb faster_rcnn_inception_resnet_v2_atrous_oi4.pb
rm -r faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12*
