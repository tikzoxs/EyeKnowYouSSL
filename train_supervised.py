from tensorflow import keras
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import os

import absl.app as app
import absl.flags as flags
import absl.logging as logging

import generators

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Where to store files.')
flags.mark_flag_as_required('workdir')

flags.DEFINE_string('train_folder', None, 'Where are the training images. Should be a folder path.')
# flags.mark_flag_as_required('train_folder')

flags.DEFINE_string('validation_folder', None, 'Where are the validating images. Should be a folder path.')
# flags.mark_flag_as_required('val_folder')

flags.DEFINE_string('test_folder', None, 'Where are the testing images. Should be a folder path.')
# flags.mark_flag_as_required('test_folder')

flags.DEFINE_string('train_batch_size', None, 'train_batch_size')
flags.mark_flag_as_required('train_batch_size')

flags.DEFINE_string('steps_per_epoch', None, 'steps_per_epoch')
flags.mark_flag_as_required('steps_per_epoch')

flags.DEFINE_string('validation_batch_size', None, 'validation_batch_size')
# flags.mark_flag_as_required('validation_batch_size')

flags.DEFINE_string('test_batch_size', None, 'test_batch_size')
# flags.mark_flag_as_required('test_batch_size')

flags.DEFINE_string('epochs', None, 'epochs')
# flags.mark_flag_as_required('epochs')

flags.DEFINE_string('tensorboard_logs_directory', None, 'TB dir')
flags.mark_flag_as_required('tensorboard_logs_directory')


def main(unused_argv):
	# logging.info('config: %s', FLAGS)
	logging.info('workdir: %s', FLAGS.workdir)
	ckpt_supervised = FLAGS.get_flag_value('workdir', None) + "/weights.h5"

	base_model = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights=ckpt_supervised, input_tensor=None, input_shape=(144,256,1), pooling='avg')
	x = base_model.output
	# x = keras.layers.Dense(1024, activation='relu')(x)
	predictions = keras.layers.Dense(4, activation='softmax')(x)

	eyeknowyou_supervised_model = keras.models.Model(inputs=base_model.input, outputs=predictions)

	eyeknowyou_supervised_model.summary()
	eyeknowyou_supervised_model.compile(optimizer=keras.optimizers.Adadelta(), 
	              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              metrics=['sparse_categorical_accuracy'])
	checkpointer = ModelCheckpoint(filepath=ckpt_supervised, verbose=1, save_best_only=True)
	reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
	                              patience=10, min_lr=0.001)
	tensorboard = keras.callbacks.TensorBoard(log_dir=FLAGS.get_flag_value('tensorboard_logs_directory', './logs'), histogram_freq=0, 
		batch_size=int(FLAGS.get_flag_value('train_batch_size', None)), write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, 
		embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
	history = eyeknowyou_supervised_model.fit_generator(
		generators.eyeknowyouTrainGenerator(), 
		steps_per_epoch=int(FLAGS.get_flag_value('steps_per_epoch', None)), epochs=int(FLAGS.get_flag_value('epochs', None)), verbose=1, callbacks=[checkpointer,tensorboard,reduce_lr], 
		validation_data=generators.eyeknowyouTrainGenerator(), validation_steps=50, validation_freq=1, class_weight=None, 
		max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

	print('\nhistory dict:', history.history)

	eyeknowyou_supervised_model.reset_metrics()

	eyeknowyou_supervised_model.save(FLAGS.get_flag_value('workdir', None))

	logging.info('I\'m done with my work, ciao!')


if __name__ == '__main__':
	app.run(main)