from model import GeneratorPretraining, Discriminator, Generator
from utils import GeneratorPretrainingGenerator, DiscriminatorGenerator
from rl import Agent, Environment
import os
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
tflag=0

class Trainer(object):
    def __init__(self, B, T, g_E, g_H, d_E, d_H, d_dropout,
              path_pos, path_neg, g_lr=1e-3, d_lr=1e-3,
              n_sample=10000, generate_samples=5, init_eps=0.1):

        self.B, self.T = B, T
        self.g_E, self.g_H = g_E, g_H
        self.d_E, self.d_H = d_E, d_H
        self.d_dropout = d_dropout
        self.generate_samples = generate_samples
        self.g_lr, self.d_lr = g_lr, d_lr
        self.eps = init_eps
        self.init_eps = init_eps
        self.top = os.getcwd()
        self.path_pos = path_pos
        self.path_neg = path_neg

        self.g_data = GeneratorPretrainingGenerator(self.path_pos, B=B, T=T,
                                                    min_count=1)  # next方法产生x, y_true数据; 都是同一个数据，比如[BOS, 8, 10, 6, 3, EOS]预测[8, 10, 6, 3, EOS]
        self.d_data = DiscriminatorGenerator(path_pos=self.path_pos, path_neg=self.path_neg, B=self.B,
                                             shuffle=True)  # next方法产生 pos数据和neg数据

        self.V = self.g_data.V



        self.agent = Agent( B, self.V, g_E, g_H, g_lr)
        self.g_beta = Agent( B, self.V, g_E, g_H, g_lr)

        self.discriminator = Discriminator(self.V, d_E, d_H, d_dropout)

        self.env = Environment(self.discriminator, self.g_data, self.g_beta, n_sample=n_sample)

        self.generator_pre = GeneratorPretraining(self.V, g_E, g_H)
        print ("para is ",self.V,g_E,g_H)

    def pre_train(self, g_epochs=3, d_epochs=1, g_pre_path=None, d_pre_path=None, g_lr=1e-3, d_lr=1e-3):
        self.pre_train_generator(g_epochs=g_epochs, g_pre_path=g_pre_path, lr=g_lr)

        self.pre_train_discriminator(d_epochs=d_epochs, d_pre_path=d_pre_path, lr=d_lr)
        print("end pretrain")

    def pre_train_generator(self, g_epochs=3, g_pre_path=None, lr=1e-3):
        if g_pre_path is None:
            self.g_pre_path = os.path.join(self.top, 'data', 'save', 'generator_pre.hdf5')
        else:
            self.g_pre_path = g_pre_path

        g_adam = keras.optimizers.Adam(lr)
        self.generator_pre.compile(g_adam, 'categorical_crossentropy')
        print('Generator pre-training')
        self.generator_pre.summary()

        self.generator_pre.fit_generator(
            self.g_data,
            steps_per_epoch=None,
            epochs=g_epochs)
        self.generator_pre.save_weights(self.g_pre_path)
        self.reflect_pre_train()

    def pre_train_discriminator(self, d_epochs=1, d_pre_path=None, lr=1e-3):
        if d_pre_path is None:
            self.d_pre_path = os.path.join(self.top, 'data', 'save', 'discriminator_pre.hdf5')
        else:
            self.d_pre_path = d_pre_path

        print('Start Generating sentences')
        #fix
        # self.agent.generator.generate_samples(self.T, self.g_data,
        #                                       self.generate_samples, self.path_neg)
        self.agent.generator.generate_samples(self.T, self.g_data,
                                              self.generate_samples, self.path_neg)
        print("generating sentences")
        self.d_data = DiscriminatorGenerator(
            path_pos=self.path_pos,
            path_neg=self.path_neg,
            B=self.B,
            shuffle=True)

        d_adam = keras.optimizers.Adam(lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.discriminator.summary()
        print('Discriminator pre-training')


        self.discriminator.fit_generator(
            self.d_data,
            steps_per_epoch=None,
            epochs=d_epochs)

        self.discriminator.save(self.d_pre_path)
        print("end dis pre_training")

    def load_pre_train(self, g_pre_path, d_pre_path):
        self.generator_pre.load_weights(g_pre_path)
        self.reflect_pre_train()
        self.discriminator.load_weights(d_pre_path)
        print("end load pre train")

    def load_pre_train_g(self, g_pre_path):
        self.generator_pre.load_weights(g_pre_path)
        self.reflect_pre_train()

    def load_pre_train_d(self, d_pre_path):
        self.discriminator.load_weights(d_pre_path)

    def reflect_pre_train(self):
        i=0
        print("relfecting")

        st = self.env.get_state()
        h, c = self.agent.generator.get_rnn_state()
        h, c = self.g_beta.generator.get_rnn_state()

        self.agent.generator(h, c, st)

        self.g_beta.generator(h, c, st)

        l_embeding=self.generator_pre.layers[1]
        l_mask = self.generator_pre.layers[2]
        l_lstm = self.generator_pre.layers[3]
        l_td=self.generator_pre.layers[4]
        #time distribute  -->dense   Necessary!
        if len(l_embeding.get_weights()) != 0:
            #have pretrained
            self.agent.generator.embedding_layer.set_weights(l_embeding.get_weights())
            self.agent.generator.mask_layer.set_weights(l_mask.get_weights())
            self.agent.generator.lstm_layer.set_weights(l_lstm.get_weights())
            self.agent.generator.dense_layer.set_weights(l_td.get_weights())

            self.g_beta.generator.embedding_layer.set_weights(l_embeding.get_weights())
            self.g_beta.generator.mask_layer.set_weights(l_mask.get_weights())
            self.g_beta.generator.lstm_layer.set_weights(l_lstm.get_weights())
            self.g_beta.generator.dense_layer.set_weights(l_td.get_weights())

        # for layer in self.generator_pre.layers:
        #     if len(layer.get_weights()) != 0:
        #         w = layer.get_weights()
        #         # print(w[0].shape)
        #         # just build a graph.
        #         main_layers=self.agent.generator.layers
        #         pre_layers=self.generator_pre.layers
        #
        #         self.agent.generator.layers[i].set_weights(w)
        #         self.g_beta.generator.layers[i].set_weights(w)
        #         i += 1
        self.agent.reset()
        self.g_beta.reset()
        self.env.reset()
        print("end reflect")
        # return


    def train(self, steps=30, g_steps=1, d_steps=1, d_epochs=1,
              g_weights_path='data/save/generator.pkl',
              d_weights_path='data/save/discriminator.hdf5',
              verbose=True,
              head=1):
        print("start adv train")
        d_adam =keras.optimizers.Adam(self.d_lr)
        # print("start adv train1")
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.eps = self.init_eps
        # print("start adv train2")
        debug_flag = 0

        log=open("data/log.txt",'w')


        for step in range(steps):

            # Generator training
            for _ in range(g_steps):
                rewards = np.zeros([self.B, self.T])
                self.agent.reset()
                self.env.reset()
                # print("start adv train4")
                global  tflag
                avg_loss=0
                for t in range(self.T):
                    state = self.env.get_state()
                    # print("start adv train5")
                    # debug_flag = 1 + debug_flag
                    # if debug_flag==2:
                    #     asdfsdfa=23
                    action = self.agent.act(state, epsilon=0.0)
                    # print("start adv train6")

                    _next_state, reward, is_episode_end, _info = self.env.step(action)
                    # print("start adv train7")
                    # print(step,_,"before update")

                    cur_loss=self.agent.generator.update(state, action, reward)
                    avg_loss+=tf.reduce_mean(cur_loss)

                    log.write("epoch" +str(step)+"g step in cur epoch "+str(t)+"loss"+str(tf.reduce_mean(cur_loss))+'\n')

                    print("epoch" ,step,"g step in cur epoch ",t,"loss",tf.reduce_mean(cur_loss))

                    # print("start adv train8")
                    rewards[:, t] = reward.reshape([self.B, ])
                    if is_episode_end:
                        if verbose:
                            print('Reward: {:.3f}, Episode end'.format(np.average(rewards)))
                            self.env.render(head=head)
                        break
                log.write("avg loss="+str(avg_loss/self.T)+'\n')
                print("avg loss=",(avg_loss/self.T))
            print("train d")
            # Discriminator training
            for _ in range(d_steps):
                self.agent.generator.generate_samples(
                    self.T,
                    self.g_data,
                    self.generate_samples,
                    self.path_neg)
                self.d_data = DiscriminatorGenerator(
                    path_pos=self.path_pos,
                    path_neg=self.path_neg,
                    B=self.B,
                    shuffle=True)
                self.discriminator.fit_generator(
                    self.d_data,
                    steps_per_epoch=None,
                    epochs=d_epochs)

            # Update env.g_beta to agent
            self.agent.save(g_weights_path)
            self.g_beta.load(g_weights_path)

            self.discriminator.save(d_weights_path)
            self.eps = max(self.eps * (1 - float(step) / steps * 4), 1e-4)

    def save(self, g_path, d_path):
        self.agent.save(g_path)
        self.discriminator.save(d_path)

    def load(self, g_path, d_path):
        self.agent.load(g_path)
        self.g_beta.load(g_path)
        self.discriminator.load_weights(d_path)

    def test(self):
        x, y = self.d_data.next()
        pred = self.discriminator.predict(x)

        for i in range(self.B):
            txt = [self.g_data.id2word[id] for id in x[i].tolist()]

            label = y[i]
            print('{}, {:.3f}: {}'.format(label, pred[i, 0], ''.join(txt)))

    def generate_txt(self, file_name, generate_samples):
        path_neg = os.path.join(self.top, 'data', 'save', file_name)

        self.agent.generator.generate_samples(
            self.T, self.g_data, generate_samples, path_neg)



