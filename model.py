import tensorflow as tf 
import numpy as np
import cv2

def load_graph(graph_pb_path):
    with open(graph_pb_path,'rb') as f:
        content = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(content)

    with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def, name='')

    return graph

def predict(graph,image):
    image_size=224
    num_channels=3
    image_array = []

    image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
    image_array.append(image)
    image_array = np.array(image_array, dtype=np.uint8)
    image_array = image_array.astype('float32')
    image_array = np.multiply(image_array-128., 1.0/128.0)

    x_batch = image_array.reshape(1, image_size,image_size,num_channels)
    y_pred = graph.get_tensor_by_name("final_result:0")
    x= graph.get_tensor_by_name("input:0")
    sess= tf.Session(graph=graph)

    feed_dict_ = {x: x_batch}
    result = sess.run(y_pred, feed_dict=feed_dict_)

    out = {
        "daisy":str(result[0][0]),
        "sunflowers":str(result[0][1]),
        "dandelion":str(result[0][2]),
        "roses":str(result[0][3]),
        "tulips":str(result[0][4])
        }

    return out
