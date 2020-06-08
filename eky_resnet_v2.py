from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras import layers

import functools

import tensorflow as tf

def bottleneck_v2(x, filters, training,  # pylint: disable=missing-docstring
                  strides=1,
                  activation_fn=tf.nn.relu,
                  normalization_fn=batch_norm,
                  kernel_regularizer=None,
                  no_shortcut=False):

  # Record input tensor, such that it can be used later in as skip-connection
  x_shortcut = x

  x = normalization_fn(x, training=training)
  x = activation_fn(x)

  # Project input if necessary
  if (strides > 1) or (filters != x.shape[-1]):
    x_shortcut = tf.keras.layers.conv2d(x, filters=filters, kernel_size=1,
                                  strides=strides,
                                  kernel_regularizer=kernel_regularizer,
                                  use_bias=False,
                                  padding='VALID')

  # First convolution
  # Note, that unlike original Resnet paper we never use stride in the first
  # convolution. Instead, we apply stride in the second convolution. The reason
  # is that the first convolution has kernel of size 1x1, which results in
  # information loss when combined with stride bigger than one.
  x = tf.keras.layers.conv2d(x, filters=filters // 4,
                       kernel_size=1,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=False,
                       padding='SAME')

  # Second convolution
  x = normalization_fn(x, training=training)
  x = activation_fn(x)
  # Note, that padding depends on the dilation rate.
  x = fixed_padding(x, kernel_size=3)
  x = tf.keras.layers.conv2d(x, filters=filters // 4,
                       strides=strides,
                       kernel_size=3,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=False,
                       padding='VALID')

  # Third convolution
  x = normalization_fn(x, training=training)
  x = activation_fn(x)
  x = tf.keras.layers.conv2d(x, filters=filters,
                       kernel_size=1,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=False,
                       padding='SAME')

  if no_shortcut:
    return x
  else:
    return x + x_shortcut

def resnet(x,  # pylint: disable=missing-docstring
           is_training,
           num_layers,
           strides=(2, 2, 2),
           num_classes=1000,
           filters_factor=4,
           weight_decay=1e-4,
           include_root_block=True,
           root_conv_size=7, root_conv_stride=2,
           root_pool_size=3, root_pool_stride=2,
           activation_fn=tf.nn.relu,
           last_relu=True,
           normalization_fn=batch_norm,
           global_pool=True,):

  end_points = {}

  filters = 16 * filters_factor

  kernel_regularizer = tf.keras.regularizers.l2(weight_decay)

  if include_root_block:
    x = fixed_padding(x, kernel_size=root_conv_size)
    x = tf.keras.layers.conv2d(x, filters=filters,
                         kernel_size=root_conv_size,
                         strides=root_conv_stride,
                         padding='VALID', use_bias=False,
                         kernel_regularizer=kernel_regularizer)

    if mode == 'v1':
      x = normalization_fn(x, training=is_training)
      x = activation_fn(x)

    x = fixed_padding(x, kernel_size=root_pool_size)
    x = tf.keras.layers.max_pooling2d(x, pool_size=root_pool_size,
                                strides=root_pool_stride, padding='VALID')
    end_points['after_root'] = x

  params = {'activation_fn': activation_fn,
            'normalization_fn': normalization_fn,
            'training': is_training,
            'kernel_regularizer': kernel_regularizer,
           }

  strides = list(strides)[::-1]
  num_layers = list(num_layers)[::-1]

  filters *= 4
  for _ in range(num_layers.pop()):
    x = bottleneck_v2(x, filters, strides=1, **params)
  end_points['block1'] = x

  filters *= 2
  x = bottleneck_v2(x, filters, strides=strides.pop(), **params)
  for _ in range(num_layers.pop() - 1):
    x = bottleneck_v2(x, filters, strides=1, **params)
  end_points['block2'] = x

  filters *= 2
  x = bottleneck_v2(x, filters, strides=strides.pop(), **params)
  for _ in range(num_layers.pop() - 1):
    x = bottleneck_v2(x, filters, strides=1, **params)
  end_points['block3'] = x

  filters *= 2
  x = bottleneck_v2(x, filters, strides=strides.pop(), **params)
  for _ in range(num_layers.pop() - 1):
    x = bottleneck_v2(x, filters, strides=1, **params)
  end_points['block4'] = x

  if (mode == 'v1') and (not last_relu):
    raise ValueError('last_relu is always True (implicitly) in the v1 mode.')

  if mode == 'v2':
    x = normalization_fn(x, training=is_training)
    if last_relu:
      x = activation_fn(x)

  if global_pool:
    x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    end_points['pre_logits'] = tf.squeeze(x, [1, 2])
  else:
    end_points['pre_logits'] = x

  if num_classes:
    logits = tf.keras.layers.conv2d(x, filters=num_classes,
                              kernel_size=1,
                              kernel_regularizer=kernel_regularizer)
    if global_pool:
      logits = tf.squeeze(logits, [1, 2])
    end_points['logits'] = logits
    eyeknowyou_ssl_model = keras.Model(inputs = x, outputs = logits)
    # return logits, end_points
    return eyeknowyou_ssl_model
  else:
    eyeknowyou_ssl_model = keras.Model(inputs = x, outputs = end_points['pre_logits'])
    # return end_points['pre_logits'], end_points
    return eyeknowyou_ssl_model

resnet50 = functools.partial(resnet, num_layers=(3, 4, 6, 3))