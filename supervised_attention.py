from tensorflow import keras
from tensorflow.keras import layers
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


class MultiHeadSelfAttention(layers.Layer):
	def __init__(self, embed_dim, num_heads=8):
		super(MultiHeadSelfAttention, self).__init__()
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		if embed_dim % num_heads != 0:
			raise ValueError(
				f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
			)
		self.projection_dim = embed_dim // num_heads
		self.query_dense = layers.Dense(embed_dim)
		self.key_dense = layers.Dense(embed_dim)
		self.value_dense = layers.Dense(embed_dim)
		self.combine_heads = layers.Dense(embed_dim)

	def attention(self, query, key, value):
		score = tf.matmul(query, key, transpose_b=True)
		dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
		scaled_score = score / tf.math.sqrt(dim_key)
		weights = tf.nn.softmax(scaled_score, axis=-1)
		output = tf.matmul(weights, value)
		return output, weights

	def separate_heads(self, x, batch_size):
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
		return tf.transpose(x, perm=[0, 2, 1, 3])

	def call(self, inputs):
		# x.shape = [batch_size, seq_len, embedding_dim]
		batch_size = tf.shape(inputs)[0]
		query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
		key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
		value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
		query = self.separate_heads(
			query, batch_size
		)  # (batch_size, num_heads, seq_len, projection_dim)
		key = self.separate_heads(
			key, batch_size
		)  # (batch_size, num_heads, seq_len, projection_dim)
		value = self.separate_heads(
			value, batch_size
		)  # (batch_size, num_heads, seq_len, projection_dim)
		attention, weights = self.attention(query, key, value)
		attention = tf.transpose(
			attention, perm=[0, 2, 1, 3]
		)  # (batch_size, seq_len, num_heads, projection_dim)
		concat_attention = tf.reshape(
			attention, (batch_size, -1, self.embed_dim)
		)  # (batch_size, seq_len, embed_dim)
		output = self.combine_heads(
			concat_attention
		)  # (batch_size, seq_len, embed_dim)
		return output

class TransformerBlock(layers.Layer):
	def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
		super(TransformerBlock, self).__init__()
		self.att = MultiHeadSelfAttention(embed_dim, num_heads)
		self.ffn = keras.Sequential(
			[layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
		)
		self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
		self.dropout1 = layers.Dropout(rate)
		self.dropout2 = layers.Dropout(rate)

	def call(self, inputs, training):
		attn_output = self.att(inputs)
		attn_output = self.dropout1(attn_output, training=training)
		out1 = self.layernorm1(inputs + attn_output)
		ffn_output = self.ffn(out1)
		ffn_output = self.dropout2(ffn_output, training=training)
		return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
	def __init__(self, maxlen, vocab_size, embed_dim):
		super(TokenAndPositionEmbedding, self).__init__()
		self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
		self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

	def call(self, x):
		maxlen = tf.shape(x)[-1]
		positions = tf.range(start=0, limit=maxlen, delta=1)
		positions = self.pos_emb(positions)
		x = self.token_emb(x)
		return x + positions

def main(unused_argv):
	ssl_embedding_dimensions = 512
	num_heads = 16
	ff_dim = 64
	# logging.info('config: %s', FLAGS)
	logging.info('workdir: %s', FLAGS.workdir)
	ckpt_supervised = FLAGS.get_flag_value('workdir', None) + "/weights.h5"

	base_model = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights=ckpt_supervised, input_tensor=None, input_shape=(144,256,1), pooling='avg')
	x = base_model.output
	x = keras.layers.Dense(512, activation='relu')(x)
	# predictions = layers.Dense(4, activation='softmax')(x)

	#Transformer layer on top of the EyeKnowYou SSL
	transformer_block = TransformerBlock(ssl_embedding_dimensions, num_heads, ff_dim)
	x = transformer_block(x)
	x = layers.GlobalAveragePooling1D()(x)
	x = layers.Dropout(0.1)(x)
	x = layers.Dense(16, activation="relu")(x)
	x = layers.Dropout(0.1)(x)
	predictions = layers.Dense(4, activation="softmax")(x)

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