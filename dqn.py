from PIL import Image
import argparse

from tf_cnnvis import *

import math
import time
import socket
import numpy as np
import sys
import tensorflow as tf
import random

import os

import random

from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data

from keras import optimizers
from keras.models import load_model
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras.api.keras.layers import PReLU, GaussianNoise, BatchNormalization, Conv2D, MaxPooling2D, Dense, Flatten, Activation, LeakyReLU

parser = argparse.ArgumentParser(description="Outsourcing playing vidya since 2018")
parser.add_argument('--load', type=str, help='Path: Filepath to the saved model')
parser.add_argument('--peek', type=str, help='Path: Filepath to the image for the layer visualizations')
parser.add_argument('--out', default='/home/tqi/work/player/out/', type=str, help='Dirpath: place to dump potential outputs')
parser.add_argument('--overfit', dest='overfit', action='store_true')
parser.add_argument('--latest', dest='load_latest', action='store_true')
parser.set_defaults(overfit=False)
parser.set_defaults(load_latest=False)

opts = parser.parse_args()

#HEIGHT = 448
#WIDTH = 384
HEIGHT = 256
WIDTH = 256

NEURONS = 150

TCP_IP = '192.168.0.16'
TCP_PORT = 8008
BUFFER_SIZE = 1024

class Communicator():
   
    def __init__(self):
        self.queue = []
        self.image = np.zeros(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.metadata = np.zeros(shape=3, dtype=np.uint32)
        self.player_choice = np.zeros(shape=1, dtype=np.uint8)
        self.conn = None

    def start_listening(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        self.conn, addr = s.accept()

    def recv_into(self, arr):
        view = memoryview(arr).cast('B')
        while len(view):
            nrecv = self.conn.recv_into(view)
            view = view[nrecv:]
   
    def download_frame(self):
        self.recv_into(self.image)
        self.recv_into(self.metadata)

    def get_player_choice(self):
        self.recv_into(self.player_choice)
        return self.player_choice[0]

    def get_data(self):
        self.download_frame()
        reward = self.metadata[0]
        menu_state = self.metadata[1]
        is_dead = self.metadata[2]
        print("metadata")
        print(self.metadata)
        sys.stdout.flush()
        return [self.image, reward, menu_state, is_dead]
        
    def cleanup(self):
        self.conn.close()
    
    def send_data(self, data):
        self.conn.send(data)

    def send_resume(self, signal=0):
        dummy = np.zeros(shape=9, dtype=np.float32)
        dummy[0] = signal
        self.send_data(dummy)

class PlayerTrainer():
    def __init__(self):
        self.communicator = Communicator()
        self.model = self.create_model()
        self.image_queue = []
        self.player_choice = []
        #1 if and only if we're about to die, 0 otherwise
        self.terminal_queue = []
        self.reward_queue = []
        self.gamma = 0.95
        #self.penalty = -1000

    #for when a new frame is sent, but its the same as the old frame but is_dead = true
    #because is_dead got updated on the same frame, but after the previous capture
    def revise_death(self):
        self.image_queue.pop()
        self.player_choice.pop()
        self.terminal_queue.pop()
        self.reward_queue.pop()
        self.terminal_queue[-1] = 1
        #not sure if this works in all edge cases, but it should be fine here
        self.reward_queue[-1] -= 1

    def update_terminal(self, terminal_reward):
        self.terminal_queue[-1] = 1
        self.reward_queue[-1] = terminal_reward
    
    def biased_sample(self, sess, n, preprocess, bias=3):
        rand_indexes = np.random.permutation(len(self.terminal_queue))
        ret = []
        num_terminals = 0
        for i in rand_indexes:
            if self.terminal_queue[i] == 0:
               keep_chance = 1.0 / bias 
               roll = random.uniform(0, 1)
               if keep_chance > roll:
                   ret.append(self.process(sess, i, preprocess))
            else:
                ret.append(self.process(sess, i, preprocess))
                num_terminals += 1
            if len(ret) == n:
                print("percent terminal is: " + str(num_terminals / n))
                sys.stdout.flush()
                return ret
        return ret

    def sample(self, n):
        rand_indexes = random.sample(range(len(self.terminal_queue)), n)
        ret = []
        for i in rand_indexes:
            ret.append(self.process(sess, i))
        return ret

    def save_last_n_images(self, n, default_name="/home/tqi/work/player/peek/capture_"):
        queue_length = len(self.terminal_queue)
        x = min(queue_length, n)
        for i in range(x):
            frame = np.reshape(self.image_queue[queue_length - 1 - i], (HEIGHT, WIDTH))
            im = Image.fromarray(frame.astype('uint8'))
            fullname = default_name + str(i) + ".bmp"
            im.save(fullname)
    
    def save_data(self):
        np.savez("overfit_data/data", 
            player_choice=np.array(self.player_choice),
            terminal_queue=np.array(self.terminal_queue),
            reward_queue=np.array(self.reward_queue)
        )

    def load_data(self):
        data = np.load("overfit_data/data.npz")
        self.player_choice = data["player_choice"].tolist()
        self.terminal_queue = data["terminal_queue"].tolist()
        self.reward_queue = data["reward_queue"].tolist()
    
    def load_images(self, num):
        base_name = "overfit_data/pictures/cap_"
        print("Loading images")
        for i in range(num + 1):
            full_name = base_name + str(i) + ".bmp"
            image = np.array(Image.open(full_name))
            reshaped = np.reshape(image, (HEIGHT, WIDTH, 1))
            self.image_queue.append(reshaped)
        print("Images loaded")
    
    def predict_q_value(self, sess, image, learning_phase=1):
        return sess.run(
            [self.model['prediction']],
            feed_dict = {
                self.model['image']: [image],
                K.learning_phase(): learning_phase
            }
        )
    
    def process(self, sess, index, preprocess):
        player_choice = self.player_choice[index]
        reward = self.reward_queue[index]
        image = self.image_queue[index]
        is_terminal = self.terminal_queue[index]

        if is_terminal:
            next_image = None
        else:
            next_image = self.image_queue[index + 1]
        
        if not preprocess:
            return (image, next_image, player_choice, reward, is_terminal, index)
        
        predicted_qs = self.predict_q_value(sess, image)[0][0]
        if self.terminal_queue[index] == 1: # we can assume the end of the queue is a terminal image
            next_image = None
            adjusted_reward = reward
            next_qs = None
        else:
            next_image = self.image_queue[index + 1]
            next_qs = self.predict_q_value(sess, next_image)[0][0]
            adjusted_reward = reward + (self.gamma * np.max(next_qs))

        target_qs = predicted_qs.copy()
        target_qs[player_choice] = adjusted_reward
        return (image, target_qs, next_qs)

    def clear_queue(self):
        self.image_queue = []
        self.player_choice = []
        self.terminal_queue = []
        self.reward_queue = []

    def maybe_prune_queue(self):
        if len(self.player_choice) > 30000:
            del self.image_queue[0:3000]
            del self.player_choice[0:3000]
            del self.terminal_queue[0:3000]
            del self.reward_queue[0:3000]
    
    def get_and_maybe_save_data(self):
        data = self.communicator.get_data()
        image = data[0]
        reward = data[1]
        #menu_state = data[2]
        is_dead = data[3]

        luminosity = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        reshaped = np.reshape(luminosity, (HEIGHT, WIDTH, 1))
       
        if not is_dead:
            self.image_queue.append(np.copy(reshaped))
            self.terminal_queue.append(0)
            self.reward_queue.append(reward)

        return (reshaped, reward, is_dead)

    def save_choice(self, c):
        self.player_choice.append(c)

    def create_model(self):
        inp = tf.placeholder(np.uint8, shape=[None, HEIGHT, WIDTH, 1], name='img_ph')
        target = tf.placeholder(np.float32, shape=[None, 9], name='target_ph')
        w_reg = lambda: None
        
        #gray = tf.image.rgb_to_grayscale(tf.to_float(tf.convert_to_tensor(inp, np.uint8)))
        #img = tf.div(gray, 255.0)

        img = tf.div(tf.to_float(tf.convert_to_tensor(inp, np.uint8)), 255.0)
        #rgb_weights = [0.2126, 0.7152, 0.0722]
        #rank_1 = tf.expand_dims(tf.rank(img) - 1, 0)
        #luminosity = tf.reduce_sum(img * rgb_weights, rank_1, keep_dims=True)
        #grayness = tf.reduce_mean(img, rank_1, keep_dims=True)
        #expanded_grayness = tf.stack([grayness, grayness, grayness], axis=-1)
        #gray_stray = tf.reduce_sum(tf.abs(img - grayness), rank_1, keep_dims=True)
        #processed = tf.clip_by_value(luminosity - 7 * gray_stray, 0.0, 1.0)

        #processed.set_shape(img.get_shape()[:-1].concatenate([1]))

        with tf.device('/gpu:0'):
            first_conv = Conv2D(filters=64, kernel_size=(8,8), padding='same', strides=[2, 2], kernel_initializer='glorot_normal', kernel_regularizer=w_reg(), input_shape=(HEIGHT, WIDTH, 1), data_format='channels_last', activation=tf.nn.relu, name='first_convolution')

            post_first_conv = first_conv(img)

            second_conv = Conv2D(filters=32, kernel_size=(8, 8), padding='same', strides=[2,2], kernel_initializer='glorot_normal', kernel_regularizer=w_reg(), activation = tf.nn.relu, name="second_convolution")

            post_second_conv = second_conv(post_first_conv)

            third_conv = Conv2D(filters=64, kernel_size=(4,4), padding='same', strides=[2,2], kernel_initializer='glorot_normal', kernel_regularizer=w_reg(), activation=tf.nn.relu, name='third_convolution')
            post_third_conv = third_conv(post_second_conv)
            
            flattened = Flatten()(post_third_conv)

            dense = Dense(256, activation = tf.nn.relu)
            post_dense = dense(flattened)
    
            prediction = Dense(9)(post_dense)

            #end inference
            
            target = tf.convert_to_tensor(target, np.float32)
        
            error = tf.reduce_sum(tf.square(prediction - target))
            train_op = tf.train.AdamOptimizer(1e-4).minimize(error)
            
            #summaries
            tf.summary.scalar('error', error)
            tf.summary.image("conv1", tf.transpose(first_conv.kernel, (3, 0, 1, 2)), max_outputs=60)
            tf.summary.scalar('conv1_max', tf.reduce_max(first_conv.kernel))
            tf.summary.scalar('conv1_min', tf.reduce_min(first_conv.kernel))
            tf.summary.scalar('conv2_max', tf.reduce_max(second_conv.kernel))
            tf.summary.scalar('conv2_min', tf.reduce_min(second_conv.kernel))
            tf.summary.scalar('conv3_max', tf.reduce_max(third_conv.kernel))
            tf.summary.scalar('conv3_min', tf.reduce_min(third_conv.kernel))
            tf.summary.scalar('dense', tf.reduce_max(dense.kernel))
            tf.summary.scalar('dense', tf.reduce_min(dense.kernel))
            #tf.summary.image("post_convs", post_third_conv, max_outputs=5)
               
            tf.summary.image('input', img, max_outputs=5)
            merged_summary = tf.summary.merge_all()

            return {
                'image': inp,
                'target': target,
                'first_conv': first_conv,
                'error': error,
                'train_op': train_op,
                'prediction': prediction,
                'summary': merged_summary,
                'conv_one_results': post_first_conv,
                'conv_two_results': post_second_conv,
                'input_tensor': img
            }
    

def save_image_from_np(frame, identifier, default_name="/home/tqi/work/player/peek/capture_"):
    im = Image.fromarray(frame)
    fullname = default_name + str(identifier) + ".bmp"
    im.save(fullname)

def save_grayscale_image_from_np(frame, identifier, default_name):
    pic = np.reshape(frame, (HEIGHT, WIDTH))
    im = Image.fromarray(pic.astype('uint8'))
    fullname = default_name + str(identifier) + ".bmp"
    im.save(fullname)

def predict_q_value(sess, model, image, learning_phase=1):
    return sess.run(
        [model['prediction']],
        feed_dict = {
            model['image']: [image],
            K.learning_phase(): learning_phase
        }
    )

def train_model(sess, model, image, target_qs, batched=False):
    if not batched:
        image = [image]
        target_qs = [target_qs]
    
    _, error, summary = sess.run(
        [model['train_op'], model['error'], model['summary']],
        feed_dict = {
            model['image']: image,
            model['target']: target_qs,
            K.learning_phase(): 1
        }
    )
    return (summary, error)

def save_model(saver, sess, iteration):
    print("Saving!")
    timestamp = datetime.today().strftime('%Y-%m-%d')
    filename = timestamp + '_iter_' + str(iteration) + '.ckpt'
    saver = tf.train.Saver(max_to_keep=10)
    saver.save(sess, ("/home/tqi/work/player/models/" + filename))

def print_rec(q_values):
    q_min = np.min(q_values)
    q_max = np.max(q_values)
    print(str(q_min) + "\t" + str(q_max))
    dots = ['⠁', '⠃', '⠇', '⠏', '⠟', '⠿', '⡿', '⣿', '⭕']
    print(q_values)
    indexes = np.argsort(q_values)
    representations = ["","","","","","","","",""] #fill it up with whatever
    for i in range(9):
        representations[indexes[i]] = dots[i]
    print(representations[8], representations[1], representations[2])
    print(representations[7], representations[0], representations[3])
    print(representations[6], representations[5], representations[4])

def visualize_and_save(sess, model, index):
    layers = ["r", "c"]
    image = np.array(Image.open("/home/tqi/work/player/peek/white_sample.bmp"))
    reshaped = np.reshape(image, (HEIGHT, WIDTH, 1))
    is_success = deconv_visualization(
        sess_graph_path = sess,
        value_feed_dict = 
            {
                model['image']: [reshaped],
                K.learning_phase(): 0
            },
        input_tensor = model['input_tensor'],
        layers = layers,
        path_outdir="player_vis/" + str(index) + "/"
    )

def run():
    batch_size = 500
    batch_counter = 0
    trainer = PlayerTrainer()
    communicator = trainer.communicator
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    tf_config.operation_timeout_in_ms = 60000

    with tf.Session(config=tf_config) as sess:
        saver = tf.train.Saver()
        if opts.peek is not None and opt.load is not None:
            #always loading latest for now
            #saver.restore(sess, opts.load)
            saver.restore(sess, tf.train.latest_checkpoint('./models/'))
            layers = ["r", "c"]
            image = np.array(Image.open(opts.peek))
            is_success = deconv_visualization(
                sess_graph_path = sess,
                value_feed_dict = 
                    {
                        model['image']: [image],
                        K.learning_phase(): 0
                    },
                input_tensor = model['input_tensor'],
                layers = layers,
                path_outdir="player_vis"
            )
            return
        elif opts.load_latest:
            saver.restore(sess, tf.train.latest_checkpoint('./models/'))
             
        train_counter = 0
        iteration = 0
        
        summary_recorder = tf.summary.FileWriter("/home/tqi/work/player/summary", sess.graph) 

        coord = tf.train.Coordinator()
        model = trainer.model
        if opts.load is not None:
            saver.restore(sess, tf.train.latest_checkpoint('./models/'))
        else:
            sess.run(tf.global_variables_initializer())
        
        #Live training
        if not opts.overfit:
            print("Listening for connection")
            communicator.start_listening()
            while True:
                np_frame, np_reward, np_is_dead = trainer.get_and_maybe_save_data()
                prediction = predict_q_value(sess, model, np_frame)[0][0]
                if np_is_dead == 0:
                    print_rec(prediction)
                    sys.stdout.flush()
                    np_prediction = np.asarray(prediction, dtype=np.float32)
                    communicator.send_data(np_prediction)
                    player_choice = communicator.get_player_choice()
                    trainer.save_choice(player_choice)
                else:
                    if np_reward < 1:
                        trainer.revise_death()
                    else:
                        trainer.update_terminal(np_reward)
                    iteration += 1
                    trained = 0
                    memory_size = len(trainer.terminal_queue)
                    print("Total frames in memory:" + str(memory_size))
                    if memory_size > batch_size:
                        data = trainer.biased_sample(sess, batch_size, True)
                        for d in data:
                            image, target_qs, next_qs = d

                            summary, error = train_model(sess, model, image, target_qs)
                            summary_recorder.add_summary(summary, train_counter)

                            train_counter += 1
                            sys.stdout.flush()

                        batch_counter += 1
                        trained = 1
                        print("Batch counter: " + str(batch_counter))
                        #trainer.save_last_n_images(100) #used to get a sample for cnnvis
                        trainer.maybe_prune_queue()

                        if batch_counter % 50 == 0:
                            save_model(saver, sess, batch_counter)
                            #trainer.save_data()
                            
                            visualize_and_save(sess, model, batch_counter)
                    communicator.send_resume(trained)
                    sys.stdout.flush()

        #Continuous training to try to overfit
        else:
            trainer.load_data()
            trainer.load_images(10009) 
            print("Entering continuous training to overfit")
            filename = "logs/big_errors.log"
            with open(filename, "w") as fh:
                while train_counter < 200000: 
                    print("Continuous training iteration" + str(train_counter))
                    sys.stdout.flush()
                    data = trainer.biased_sample(sess, batch_size, False)
                    for d in data:
                        image, next_image, player_choice, reward, is_terminal, img_index = d
                        if is_terminal:
                            adjusted_score = reward
                        else:
                            next_best_q = np.max(predict_q_value(sess, model, next_image)[0][0])
                            half_maximum_cumulative_reward = 30 * 60 / 2
                            adjusted_score = reward + min((trainer.gamma * next_best_q), half_max_cumulative_reward)
                            
                        prediction = predict_q_value(sess, model, image)[0][0]
                        
                        if not is_terminal:
                            next_prediction = predict_q_value(sess, model, next_image)[0][0]
                        else:
                            next_prediction = None

                        target_qs = prediction.copy()
                        target_qs[player_choice] = adjusted_score

                        summary, error = train_model(sess, model, image, target_qs)
                        summary_recorder.add_summary(summary, train_counter)
                        
                        prediction_post = predict_q_value(sess, model, image)[0][0]
                        if not is_terminal:
                            next_prediction_post = predict_q_value(sess, model, next_image)[0][0]
                        else:
                            next_prediction = None

                        train_counter += 1
                        sys.stdout.flush()
                        if error > 50:
                            print("Extreme case detected")
                            sys.stdout.flush()
                            fh.write("Iteration: {}, FileIndex: {}, Error: {}, Terminal: {}\n".format(train_counter, img_index, error, is_terminal))
                            fh.write("Target: {}, Index: {}\n".format(adjusted_score, player_choice))
                            fh.write("Before And After Indexed Prediction: {}, {}\n".format(prediction[player_choice], prediction_post[player_choice]))

                            fh.write("Current Pre Prediction\n")
                            fh.write(np.array2string(prediction))
                            fh.write("\nCurrent Post Prediction\n")
                            fh.write(np.array2string(prediction_post))
                                
                            save_grayscale_image_from_np(image, img_index, "/home/tqi/work/player/logs/image_")
                            if not is_terminal:
                                save_grayscale_image_from_np(next_image, img_index + 1, "/home/tqi/work/player/logs/image_")
                                fh.write("\nMax Before And After Indexed Prediction: {}, {}\n".format(np.max(next_prediction), np.max(next_prediction_post)))
                                fh.write("Next Pre Prediction\n")
                                fh.write(np.array2string(next_prediction))
                                fh.write("\nNext Post Prediction\n")
                                fh.write(np.array2string(next_prediction_post))
                            fh.write("\n\n\n")

                    batch_counter += 1
                    trained = 1
                    #trainer.save_last_n_images(100) // used to get a sample for cnnvis
                    trainer.maybe_prune_queue()

                    if batch_counter % 50 == 0:
                        save_model(saver, sess, batch_counter)
                        #trainer.save_data()

                        visualize_and_save(sess, model, batch_counter)

            pic_index = 0
            trainer.save_data()
            for pic in trainer.image_queue:
                save_grayscale_image_from_np(pic, pic_index, "/home/tqi/work/player/overfit_data/pictures/cap_")
                pic_index += 1
            communicator.send_resume(trained)
            print("Done")
            return
        communicator.cleanup()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
run()
