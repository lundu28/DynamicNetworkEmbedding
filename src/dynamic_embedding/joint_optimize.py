import numpy as np
import tensorflow as tf
import math

from utils.data_handler import DataHandler as dh

RANDOM_SEED = 157
class NodeEmbedding(object):
    def __init__(self, params, init_embeddings, init_weights, G, mapp, rmapp, num_new = 1):
        self.num_nodes_init, self.embedding_size = init_embeddings.shape
        self.num_nodes = self.num_nodes_init + num_new

        self.num_modify = params["num_modify"]
        self.batch_size = params["batch_size"]
        self.learn_rate = params["learn_rate"]
        self.optimizer = params["optimizer"] if "optimizer" in params else "GradientDescentOptimizer"
        self.tol = params["tol"] if "tol" in params else 0.0001
        self.neighbor_size = params["neighbor_size"]
        self.negative_distortion = params["negative_distortion"]
        self.num_sampled = params["num_sampled"]
        self.epoch_num = params["iteration_num"]
        self.lbd = params["lambda"]

        self.bs = __import__("batch_strategy." + params["batch_strategy"]["func"], fromlist = ["batch_strategy"]).BatchStrategy(G, num_new, mapp, rmapp, self.num_modify, params["batch_strategy"])

        unigrams_in = None
        if "in_negative_sampling_distribution" in params:
            unigrams_in = getattr(dh, params["in_negative_sampling_distribution"]["func"])(G, params["in_negative_sampling_distribution"])
        unigrams_out = None
        if "out_negative_sampling_distribution" in params:
            unigrams_out = getattr(dh, params["out_negative_sampling_distribution"]["func"])(G, params["out_negative_sampling_distribution"])

        self.tensor_graph = tf.Graph()

        with self.tensor_graph.as_default():
            tf.set_random_seed(RANDOM_SEED)
            self.constant_embeddings = tf.constant(init_embeddings[: self.num_nodes_init - self.num_modify])
            self.constant_weights = tf.constant(init_weights[: self.num_nodes_init - self.num_modify])
            self.modify_embeddings_pre = tf.constant(init_embeddings[self.num_nodes_init - self.num_modify :])
            self.modify_weights_pre = tf.constant(init_weights[self.num_nodes_init - self.num_modify :])
            self.modify_embeddings = tf.Variable(tf.random_uniform([self.num_modify, self.embedding_size], -1.0, 1.0), dtype = tf.float32)
            self.modify_weights = tf.Variable(tf.random_uniform([self.num_modify, self.embedding_size], -1.0, 1.0), dtype = tf.float32)


            self.x_in = tf.placeholder(tf.int64, shape = [None])
            self.x_out = tf.placeholder(tf.int64, shape = [None])
            self.labels_in = tf.placeholder(tf.int64, shape = [None, self.neighbor_size])
            self.labels_out = tf.placeholder(tf.int64, shape = [None, self.neighbor_size])

            self.new_embeddings = tf.Variable(tf.random_uniform([num_new, self.embedding_size], -1.0, 1.0), dtype = tf.float32)
            self.new_weights = tf.Variable(tf.random_uniform([num_new, self.embedding_size], -1.0, 1.0), dtype = tf.float32)
            self.nce_biases = tf.zeros([self.num_nodes], tf.float32)

            self.embed = tf.concat([self.constant_embeddings, self.modify_embeddings, self.new_embeddings], 0)
            self.w = tf.concat([self.constant_weights, self.modify_weights, self.new_weights], 0)

            self.delta_embeddings_pad = tf.concat(
                    [tf.zeros([self.num_nodes_init - self.num_modify, self.embedding_size], dtype = tf.float32),
                        self.modify_embeddings - self.modify_embeddings_pre,
                        tf.zeros([num_new, self.embedding_size], dtype = tf.float32)],
                    axis = 0)
            self.delta_weights_pad = tf.concat(
                    [tf.zeros([self.num_nodes_init - self.num_modify, self.embedding_size], dtype = tf.float32),
                        self.modify_weights - self.modify_weights_pre,
                        tf.zeros([num_new, self.embedding_size], dtype = tf.float32)],
                    axis = 0)

            self.embedding_batch = tf.nn.embedding_lookup(self.embed, self.x_in)
            self.weight_batch = tf.nn.embedding_lookup(self.w, self.x_out)
            self.delta_embeddings_batch = tf.nn.embedding_lookup(self.delta_embeddings_pad, self.x_in)
            self.delta_weights_batch = tf.nn.embedding_lookup(self.delta_weights_pad, self.x_out)

            if unigrams_in is None:
                self.loss_in = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights = self.w,
                        biases = self.nce_biases,
                        labels = self.labels_in,
                        inputs = self.embedding_batch,
                        num_sampled = self.num_sampled,
                        num_classes = self.num_nodes,
                        num_true = self.neighbor_size))
            else:
                self.sampled_values_in = tf.nn.fixed_unigram_candidate_sampler(
                    true_classes = self.labels_in,
                    num_true = self.neighbor_size,
                    num_sampled = self.num_sampled,
                    unique = False,
                    range_max = self.num_nodes,
                    distortion = self.negative_distortion,
                    unigrams = unigrams_in)
                self.loss_in = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights = self.w,
                        biases = self.nce_biases,
                        labels = self.labels_in,
                        inputs = self.embedding_batch,
                        num_sampled = self.num_sampled,
                        num_classes = self.num_nodes,
                        num_true = self.neighbor_size,
                        sampled_values = self.sampled_values_in))

            if unigrams_out is None:
                self.loss_out = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights = self.embed,
                        biases = self.nce_biases,
                        labels = self.labels_out,
                        inputs = self.weight_batch,
                        num_sampled = self.num_sampled,
                        num_classes = self.num_nodes,
                        num_true = self.neighbor_size))
            else:
                self.sampled_values_out = tf.nn.fixed_unigram_candidate_sampler(
                    true_classes = self.labels_out,
                    num_true = self.neighbor_size,
                    num_sampled = self.num_sampled,
                    unique = False,
                    range_max = self.num_nodes,
                    distortion = self.negative_distortion,
                    unigrams = unigrams_out)
                self.loss_out = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights = self.embed,
                        biases = self.nce_biases,
                        labels = self.labels_out,
                        inputs = self.weight_batch,
                        num_sampled = self.num_sampled,
                        num_classes = self.num_nodes,
                        num_true = self.neighbor_size,
                        sampled_values = self.sampled_values_out))

            self.loss = self.loss_out + self.loss_in + self.lbd * (tf.norm(self.delta_embeddings_batch) + tf.norm(self.delta_weights_batch))
            self.train_step = getattr(tf.train, self.optimizer)(self.learn_rate).minimize(self.loss)

    def train(self, save_path = None):
        print("[] Start a new turn of embedding...")
        with tf.Session(graph = self.tensor_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in xrange(self.epoch_num):
                batch_x_in, batch_x_out, batch_labels_in, batch_labels_out = self.bs.get_batch(self.batch_size)
                self.train_step.run({
                    self.x_in : batch_x_in,
                    self.x_out : batch_x_out,
                    self.labels_in : batch_labels_in,
                    self.labels_out : batch_labels_out})
                if (i % 1000 == 0):
                    loss = self.loss.eval({
                        self.x_in : batch_x_in,
                        self.x_out : batch_x_out,
                        self.labels_in : batch_labels_in,
                        self.labels_out : batch_labels_out})
                    print("Iteration %d, loss: %f" % (i, loss))
            print("[+] Finished a new turn of embedding!")
            return sess.run(self.embed), sess.run(self.w)

