#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json, math, os
from absl import app
from absl import flags
import resnet
import data as data_lib
import model as model_lib
import model_util as model_util
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from pprint import pprint
import torch, cv2
import torch.nn as nn
from resnet_pytorch import resnet50
import numpy as np
import os.path as osp


FLAGS = flags.FLAGS


flags.DEFINE_float('learning_rate', 0.3, 'Initial learning rate per batch size of 256.')
flags.DEFINE_float('warmup_epochs', 10, 'Number of epochs of warmup.')
flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')
flags.DEFINE_float('batch_norm_decay', 0.9, 'Batch norm decay parameter.')
flags.DEFINE_integer('train_batch_size', 512, 'Batch size for training.')
flags.DEFINE_string('train_split', 'train', 'Split for training.')
flags.DEFINE_integer('train_epochs', 100, 'Number of epochs to train for.')
flags.DEFINE_integer('train_steps', 0, 'Number of steps to train for. If provided, overrides train_epochs.')
flags.DEFINE_integer('eval_batch_size', 256, 'Batch size for eval.')
flags.DEFINE_integer('train_summary_steps', 100, 'Steps before saving training summaries. If 0, will not save.')
flags.DEFINE_integer('checkpoint_epochs', 1, 'Number of epochs between checkpoints/summaries.')
flags.DEFINE_integer('checkpoint_steps', 0, 'Number of steps between checkpoints/summaries. If provided, overrides checkpoint_epochs.')
flags.DEFINE_string('eval_split', 'validation', 'Split for evaluation.')
flags.DEFINE_string('dataset', 'imagenet2012', 'Name of a dataset.')
flags.DEFINE_bool('cache_dataset', False,'Whether to cache the entire dataset in memory. If the dataset is ImageNet, this is a very bad idea, but for smaller datasets it can improve performance.')
flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'train_then_eval'], 'Whether to perform training or evaluation.')
flags.DEFINE_enum('train_mode', 'pretrain', ['pretrain', 'finetune'], 'The train mode controls different objectives and trainable components.')
flags.DEFINE_string('checkpoint', './ckpts/ResNet50_4x', 'Loading from the given checkpoint for continued training or fine-tuning.')
flags.DEFINE_string('variable_schema', '?!global_step', 'This defines whether some variable from the checkpoint should be loaded.')
flags.DEFINE_bool('zero_init_logits_layer', False, 'If True, zero initialize layers after avg_pool for supervised learning.')
flags.DEFINE_integer('fine_tune_after_block', -1, 'The layers after which block that we will fine-tune. -1 means fine-tuning everything. 0 means fine-tuning after stem block. 4 means fine-tuning just the linera head.')
flags.DEFINE_string('master', None,'Address/name of the TensorFlow master to use. By default, use an in-process master.')
flags.DEFINE_string('model_dir', 'output', 'Model directory for training.')
flags.DEFINE_string('data_dir', './datasets','Directory where dataset is stored.')
flags.DEFINE_bool('use_tpu', False,'Whether to run on TPU.')
tf.flags.DEFINE_string('tpu_name', None,'The Cloud TPU to use for training. This should be either the name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
tf.flags.DEFINE_string('tpu_zone', None,'[Optional] GCE zone where the Cloud TPU is located in. If not specified, we will attempt to automatically detect the GCE project from metadata.')
tf.flags.DEFINE_string('gcp_project', None,'[Optional] Project name for the Cloud TPU-enabled project. If not specified, we will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_enum('optimizer', 'lars', ['momentum', 'adam', 'lars'],'Optimizer to use.')
flags.DEFINE_float('momentum', 0.9,'Momentum parameter.')
flags.DEFINE_string('eval_name', None, 'Name for eval.')
flags.DEFINE_integer('keep_checkpoint_max', 5, 'Maximum number of checkpoints to keep.')
flags.DEFINE_integer('keep_hub_module_max', 1, 'Maximum number of Hub modules to keep.')
flags.DEFINE_float('temperature', 0.1, 'Temperature parameter for contrastive loss.')
flags.DEFINE_boolean('hidden_norm', True, 'Temperature parameter for contrastive loss.')
flags.DEFINE_enum('head_proj_mode', 'nonlinear', ['none', 'linear', 'nonlinear'], 'How the head projection is done.')
flags.DEFINE_integer('head_proj_dim', 128, 'Number of head projection dimension.')
flags.DEFINE_integer('num_nlh_layers', 1, 'Number of non-linear head layers.')
flags.DEFINE_boolean('global_bn', True, 'Whether to aggregate BN statistics across distributed cores.')
flags.DEFINE_integer('width_multiplier', 1, 'Multiplier to change width of network.')
flags.DEFINE_integer('resnet_depth', 50, 'Depth of ResNet.')
flags.DEFINE_integer('image_size', 224, 'Input image size.')
flags.DEFINE_float('color_jitter_strength', 1.0, 'The strength of color jittering.')
flags.DEFINE_boolean('use_blur', True, 'Whether or not to use Gaussian blur for augmentation during pretraining.')


def load_conv(pth_module, tf_vars, tf_name):
    src = pth_module.weight
    tgt = tf_vars[os.path.join(tf_name, 'kernel')].transpose((3,2,0,1))
    src.data.copy_(torch.from_numpy(tgt).double())
    if pth_module.bias is not None:
        src = pth_module.bias
        tgt = tf_vars[os.path.join(tf_name, 'bias')]
        src.data.copy_(torch.from_numpy(tgt).double())
    # print(src.shape, tgt.shape)
    

def load_bn(pth_module, tf_vars, tf_name):
    src = pth_module.running_mean
    tgt = tf_vars[os.path.join(tf_name, 'moving_mean')]
    src.data.copy_(torch.from_numpy(tgt).double())
    src = pth_module.running_var
    tgt = tf_vars[os.path.join(tf_name, 'moving_variance')]
    src.data.copy_(torch.from_numpy(tgt).double())
    src = pth_module.weight
    tgt = tf_vars[os.path.join(tf_name, 'gamma')]
    src.data.copy_(torch.from_numpy(tgt).double())
    src = pth_module.bias
    tgt = tf_vars[os.path.join(tf_name, 'beta')]
    src.data.copy_(torch.from_numpy(tgt).double())
    # print(src.shape, tgt.shape)


def load_fc(pth_module, tf_vars, tf_name):
    src = pth_module.weight
    tgt = tf_vars[os.path.join(tf_name, 'kernel')]
    src.data.copy_(torch.from_numpy(tgt.transpose()).double())
    if pth_module.bias is not None:
        src = pth_module.bias
        tgt = tf_vars[os.path.join(tf_name, 'bias')]
        src.data.copy_(torch.from_numpy(tgt).double())


def build_model_fn(model, num_classes, num_train_examples):
    def model_fn(features, labels, mode, params=None):
        with tf.variable_scope('base_model'):
            hiddens = model(features, is_training=False)
        hiddens = model_util.supervised_head(hiddens, num_classes, False)
        return tf.estimator.tpu.TPUEstimatorSpec(mode=mode, predictions=hiddens)
    return model_fn


def main(argv):
    ckpt_path = './ResNet50_1x/model.ckpt-225206'

    tf_path = os.path.abspath(ckpt_path)  # Path to our TensorFlow checkpoint
    # tf_vars = tf.train.list_variables(tf_path)
    # pprint(tf_vars)
    init_vars = tf.train.list_variables(tf_path)
    tf_vars = {}
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        tf_vars[name] = array

    net = resnet50(pretrained=False)

    # STEM
    load_conv(net.conv1, tf_vars, 'base_model/conv2d/')
    load_bn(net.bn1, tf_vars, 'base_model/batch_normalization/')
    load_fc(net.fc, tf_vars, 'head_supervised/linear_layer/dense')

    current_index = 1
    layers = [3, 4, 6, 3]
    for i in range(len(layers)):
        block = getattr(net, 'layer%d'%(i+1))
        for j in range(layers[i]):
            pointer = block[j]
            if j == 0:
                # print(pointer)
                tf_name = 'base_model/conv2d_%d'%current_index
                load_conv(pointer.downsample[0], tf_vars, tf_name)
                tf_name = 'base_model/batch_normalization_%d'%current_index
                load_bn(pointer.downsample[1], tf_vars, tf_name)
                current_index += 1
            for k in range(1, 4):
                tf_name = 'base_model/conv2d_%d'%current_index
                load_conv(getattr(pointer, 'conv%d'%k), tf_vars, tf_name)
                tf_name = 'base_model/batch_normalization_%d'%current_index
                load_bn(getattr(pointer, 'bn%d'%k), tf_vars, tf_name)
                current_index += 1
    torch.save(net.state_dict(), 'simclr_resnet50.pth')

    
    inputs = cv2.resize(cv2.imread('000019.jpg', cv2.IMREAD_COLOR), (224,224)).astype(np.float) - 128
    net.eval()
    pth_y = net(torch.from_numpy(inputs.transpose((2, 0, 1))).unsqueeze(0).float()).squeeze(0).cpu().data.numpy()
    print(pth_y.shape)

    num_examples = 16
    batch_size = FLAGS.train_batch_size
    checkpoint_steps = 10
    run_config = tf.estimator.tpu.RunConfig(
        tpu_config=tf.estimator.tpu.TPUConfig(iterations_per_loop=checkpoint_steps, eval_training_input_configuration=tf.estimator.tpu.InputPipelineConfig.PER_HOST_V1),
        model_dir='./ResNet50_1x',
        save_summary_steps=checkpoint_steps,
        save_checkpoints_steps=checkpoint_steps,
        keep_checkpoint_max=10,
        master=None,
        cluster=None)
    estimator = tf.estimator.tpu.TPUEstimator(
        build_model_fn(resnet.resnet_v1(resnet_depth=50, width_multiplier=1, cifar_stem=False), 1000, num_examples),
        config=run_config,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        predict_batch_size=batch_size,
        use_tpu=False)
    
    def _input_fn(params):
        def map_fn(image, label):
            return image, label, 1.0

        images = tf.convert_to_tensor(np.stack([inputs] * num_examples, 0))
        images = tf.cast(images, tf.float32)
        labels = tf.convert_to_tensor(np.zeros((num_examples,), dtype=int))
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(map_fn)
        dataset = dataset.batch(params['batch_size'])
        xx, yy, mask = tf.data.make_one_shot_iterator(dataset).get_next()

        return xx, {'labels': yy, 'mask': mask}


    for item in estimator.predict(input_fn=_input_fn, checkpoint_path=ckpt_path):
        print(np.amax(item), np.amax(pth_y))
        # max_absolute_diff = np.amax(np.abs(item - pth_y.transpose(1,2,0)))

        max_absolute_diff = np.amax(np.abs(item - pth_y))
        print('item', item.shape, max_absolute_diff, np.argmax(item), np.argmax(item) - np.argmax(pth_y))
        print('------')


if __name__ == '__main__':
    tf.disable_eager_execution()  # Disable eager mode when running with TF2.
    app.run(main)

