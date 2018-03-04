import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

# Image augmentation utilities
class DataAug:
    def __init__(self):
#        self._aug_prob = [1.0, 1.0, 0.0]
        self._aug_prob = [0.5, 0.5, 0.5]
        self._aug_func = [self._flip_img, self._shadow_img, self._blur_img]
    def do(self, img_batch, label_mask_batch):
        rtn_img_batch = np.copy(img_batch)
        rtn_label_mask_batch = np.copy(label_mask_batch)
        for i in range(len(self._aug_prob)):
            is_aug = random.uniform(0.0,1.0)<self._aug_prob[i]
            if is_aug:
                for sample_idx, single_img in enumerate(rtn_img_batch):
                    rtn_img_batch[sample_idx,...] = self._aug_func[i](single_img)
                if self._aug_func[i]==self._flip_img:
                    for sample_idx, single_label_mask in enumerate(rtn_label_mask_batch):
                        rtn_label_mask_batch[sample_idx,...] = self._flip_img(single_label_mask)
        return rtn_img_batch, rtn_label_mask_batch
    
    # Image shadowing, (h,w,channel)
    # This function is referenced from: https://github.com/windowsub0406/SelfDrivingCarND/blob/master/SDC_project_3/model.ipynb
    def _shadow_img(self,image, min_alpha=0.5, max_alpha = 0.75):
        """generate random shadow in random region"""
        rows, cols, _ = image.shape
        top_x, bottom_x = np.random.randint(0, cols, 2)
        
        shadow_img = image.copy()
        coin = np.random.uniform()
        if coin>0.5:
            vertices = np.array([[top_x, 0], [cols, 0], [cols, rows], [bottom_x, rows]], dtype=np.int32)
        else:
            vertices = np.array([[top_x, 0], [0, 0], [0, rows], [bottom_x, rows]], dtype=np.int32)
        mask = image.copy()
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (0,) * channel_count
        cv2.fillPoly(mask, [vertices], ignore_mask_color)
        rand_alpha = np.random.uniform(min_alpha, max_alpha)
        cv2.addWeighted(mask, rand_alpha, image, 1 - rand_alpha, 0., shadow_img)
        return shadow_img
    
    # Image horizontal shifting, (h,w,channel)
    def _hshift_img(self, img):
        max_shift_range=100
        # if tr_x > 0, the image will shift right
        tr_x = int(np.random.uniform(max_shift_range*2)-max_shift_range)
        tr_y=0
        trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
        rows,cols = img.shape[:2]
        img = cv2.warpAffine(img,trans_M,(cols,rows))
        return img
    
    # Image Flipping, (h,w,channel)
    def _flip_img(self, img):
        return img[:,-1::-1,:]
    
    # Image Brightness
    def _brighten_img(self, img,low_ratio=0.5,up_ratio=1.2):
        hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        ratio = random.uniform(low_ratio,up_ratio)
        hsv[:,:,-1] = np.clip(hsv[:,:,-1]*ratio,a_min=0,a_max=255)
        return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    # Image Blurring
    def _blur_img(self, img, k=5):
        return cv2.GaussianBlur(img, (k, k), 0)

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    img_in = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return img_in, keep, l3out, l4out, l7out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output, (?, 20, 72, 256)
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output, (?, 10, 36, 512)
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output, (?, 5, 18, 4096)
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # conv_1x1: (?, 5, 18, 1024)
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, 1024, 1, padding='same',
                                activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # conv2d_transpose(input, out-dim, kernel_size, strides, ...)
    # output will have 2 times of size comparing to input
    # output: (?, 10, 36, 512)
    output = tf.layers.conv2d_transpose(conv_1x1, 512, 4, 2, padding='same',
                                        activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # skip Connections
    vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='vgg_layer4_out_scaled')
    output = tf.add(output,vgg_layer4_out_scaled)
    
    # output will have 2 times of size comparing to input
    # output: (?, 20, 72, 512)
    output = tf.layers.conv2d_transpose(output, 256, 4, 2, padding='same',
                                        activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # skip Connections
    vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='vgg_layer3_out_scaled')
    output = tf.add(output,vgg_layer3_out_scaled)
    
    # input, out-dim, kernel_size, strides, padding,
    # output will have 8 times of size comparing to input
    # output: (?, 160, 576, num_classes)
    # this layer of output tensor should suppose to have same shape with original input image
    output = tf.layers.conv2d_transpose(output, num_classes, 16, 8, padding='same',
                                        activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes, trainable_collection):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    logits = tf.reshape(nn_last_layer,(-1,num_classes))
    correct_label = tf.reshape(correct_label,(-1,num_classes))
    # xent loss
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=correct_label,logits=logits)
#    # iou
#    inter=tf.reduce_sum(tf.multiply(logits, correct_label))
#    union=tf.reduce_sum(tf.subtract(tf.add(logits,correct_label),tf.multiply(logits,correct_label)))
#    iou=tf.divide(inter,union)
##    iou_loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.divide(inter,union))
    
    # sum of xent, iou, and reg losses
    sum_of_losses = cross_entropy_loss + sum(reg_losses)

    loss_operation = tf.reduce_mean(sum_of_losses)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    if len(trainable_collection)==0:
        train_op = optimizer.minimize(loss_operation)
    else:
        print('Only update variables in decoder, num_of_vars=',len(trainable_collection))
        train_op = optimizer.minimize(loss_operation, var_list = trainable_collection)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    dataAug = DataAug()
    # NOTE: The VGG model is not frozen! we should only update the decoder part of the variables!
    lr = 0.001  # init learning rate
    accumulate_time = 0
    sess.run(tf.global_variables_initializer())
    loss_list = []
    for itr in range(epochs):
        lr_reduce_factor = (itr//10)*5 + 1
        stime = time.time()
        loss_per_epoch = 0.0
        num_samples = 0
        num_pixels = 160*576
        for img,label_mask in get_batches_fn(batch_size):
            # data augmentation
            img, label_mask = dataAug.do(img,label_mask)
            num_samples += len(label_mask)
            label = np.array(label_mask,dtype=float)
            _, loss = sess.run([train_op, cross_entropy_loss],feed_dict={input_image:img, correct_label:label, keep_prob:0.8, learning_rate:lr/lr_reduce_factor})
            loss_per_epoch += np.sum(loss)
        loss_per_epoch /= (num_samples*num_pixels)
        loss_list.append(loss_per_epoch)
        etime = time.time()
        accumulate_time += etime - stime
        print("EPOCH {} with {} training time...".format(itr+1,etime - stime))
        print("\t ",loss_per_epoch)
    return loss_list
tests.test_train_nn(train_nn)

def checkVGGShape(sess, l3out, l4out, l7out, get_batches_fn, input_image, correct_label, keep_prob, learning_rate):
    # check tensor shapes
    shape_op3 = tf.shape(l3out)
    shape_op4 = tf.shape(l4out)
    shape_op7 = tf.shape(l7out)
    sess.run(tf.global_variables_initializer())
    for img,label_mask in get_batches_fn(2):
        label = np.zeros_like(label_mask,dtype=float)
        label[label_mask] = 1.0
        print_results3, print_results4, print_results7 = sess.run([shape_op3, shape_op4, shape_op7],feed_dict={input_image:img, correct_label:label, keep_prob:0.8, learning_rate:0.0001})
        print(print_results3, "; ", print_results4, "; ", print_results7)
        break

def run(trained_model_path, epochs, batch_size):
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
#    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    if not os.path.isdir('./models'):
        os.makedirs('./models')
    
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, l3out, l4out, l7out = load_vgg(sess,vgg_path)
#        vars_encoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        with tf.variable_scope('decoder'):
            layer_output = layers(l3out, l4out, l7out, num_classes)
        vars_decoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='decoder')

        # create tf.placeholder for correct_label and learning_rate
        correct_label = tf.placeholder(tf.float32, (None, *image_shape, num_classes), name='correct_label')   # one-hot like label
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes, vars_decoder)
#        logits, iou, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes, vars_decoder)
        
        # Check VGG output shape
#        checkVGGShape(sess, l3out, l4out, l7out, get_batches_fn, input_image, correct_label, keep_prob, learning_rate)

        # TODO: Train NN using the train_nn function
        saver = tf.train.Saver(tf.global_variables())
        if not os.path.isfile(trained_model_path+'.index'):
            loss = train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)
            np.save(trained_model_path+'.npy',loss)
            
            saver.save(sess, trained_model_path)
            print("Model saved")
        else:
            loss = np.load(trained_model_path+'.npy')
            saver.restore(sess, trained_model_path)
            print("Model loaded")

        plt.figure()
        plt.plot(np.arange(len(loss))+1,loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')            
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    epochs = 300
    batch_size = 4
    trained_model_path = './models/semantic_model_epoch_{}'.format(epochs)
    run(trained_model_path, epochs, batch_size)
