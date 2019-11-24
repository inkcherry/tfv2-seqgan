import numpy as np
import tensorflow as tf
import  tensorflow.keras as keras
import random
import  pickle
# from tensorflow.keras.utils import to_categorical

def GeneratorPretraining(vocabrary_size,emb_dim, hidden_dim,):
    input = keras.Input(shape=(None,), dtype='int32', name='Input') # (B, T)
    out = keras.layers.Embedding(vocabrary_size, emb_dim,  name='Embedding')(input) # (B, T, E)
    out=keras.layers.Masking(mask_value=0,name="masking")(out)
    out = keras.layers.LSTM(hidden_dim, return_sequences=True, name='LSTM')(out)  # (B, T, H)
    out = keras.layers.TimeDistributed(
        keras.layers.Dense(vocabrary_size, activation='softmax', name='DenseSoftmax'),
        name='TimeDenseSoftmax')(out)    # (B, T, V)
    generator_pretraining = keras.models.Model(input, out)
    return generator_pretraining


class Generator(keras.Model):
    def __init__(self, batch_size,vocabrary_size, emb_dim, hidden_dim,
                 learning_rate=0.01, reward_gamma=0.95):
        super(Generator,self).__init__()
        #h-para
        self.vocabrary_size = vocabrary_size
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []
        self.temperature = 1.0
        self.grad_clip = 5.0
        self.pro_nptype = np.zeros([self.batch_size,self.vocabrary_size])
        # self.is_first_call=1
        #layers
        # self.layers=[]

        self.embedding_layer = keras.layers.Embedding(self.vocabrary_size,emb_dim, name='Embedding')
        #mask_zero will raise cudnn error
        # cudnn kernel bug ,mask_zero =true will get a error,
        self.mask_layer=keras.layers.Masking(mask_value=0,name="masking")
        self.lstm_layer      = keras.layers.LSTM(self.hidden_dim,return_state=True,name='LSTM')

        self.dense_layer =keras.layers.Dense(vocabrary_size,activation='softmax',name='DenseSoftmax')

        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        # self.layers.append(self.embedding_layer)
        # self.layers.append(self.lstm_layer)
        # self.layers.append(self.dense_layer)



        self.reset_rnn_state()





    def __call__(self,h_in,c_in,input_state) :

        x=self.embedding_layer(input_state)
        # print("here is in __call__,the shape of h_in and c_in x " ,h_in.shape,c_in.shape,x.shape)
        # print("here is in __call__,the type of h_in and c_in x ",type(h_in),type(c_in),type(x) )
        # print("here is real input")
        # print("x is ")
        # print(x)
        # print("h is")
        # print(h_in)
        # print("c_is")
        # print(c_in)

        # h_in=tf.convert_to_tensor(h_in,tf.float32)
        # c_in=tf.convert_to_tensor(c_in,tf.float32)
        out= self.mask_layer(x)
        # qqq=np.array(out)
        # if (np.isnan(qqq.any())):
        #     print("fsdf")

        af,self.next_h,self.next_c=self.lstm_layer(out,initial_state=[h_in, c_in])

        # if np.isnan(self.next_c[0][0]) :
        #     asdff=234

        prob=self.dense_layer(af)
        # if (np.isnan(prob[0][0])):
        #     print("fsdf")
        # prob=prob
        # self.cur_h=self.next_h
        # self.cur_c=self.next_c


        return prob


    def reset_rnn_state(self):
        self.h = np.zeros([self.batch_size, self.hidden_dim],dtype=np.float32)
        self.c = np.zeros([self.batch_size, self.hidden_dim],dtype=np.float32)

        self.h=tf.convert_to_tensor(self.h,tf.float32)
        self.c=tf.convert_to_tensor(self.c,tf.float32)

        # self.h=np.random.sample([self.batch_size, self.hidden_dim])
        # self.c= np.random.sample([self.batch_size, self.hidden_dim])

        # self.h = np.zeros([self.batch_size, self.hidden_dim])
        # self.c = np.zeros([self.batch_size, self.hidden_dim])

    def set_rnn_state(self, h, c):
        '''
        # Arguments:
            h: np.array, shape = (B,H)
            c: np.array, shape = (B,H)
        '''
        self.h= h
        self.c = c


    def get_rnn_state(self):
        return self.h, self.c


    def update(self, state, action, reward, h=None, c=None, stateful=True):
        # return
        if h is None:
            h = self.h
        if c is None:
            c = self.c
        state = state[:, -1].reshape(-1, 1)
        reward = reward.reshape(-1)
        action=tf.keras.utils.to_categorical(action, self.vocabrary_size)
        with tf.GradientTape() as tape:
            prob=self.__call__(h,c,state)

            # log_prob = tf.math.log(tf.reduce_mean(prob * action, axis=-1))  # (B, )


            log_prob = tf.math.log(tf.clip_by_value(tf.reduce_mean(prob * action, axis=-1),1e-10,10000) ) # (B, )
            #1e-20 will get nan loss
            # log_prob=tf.math.log(tf.reduce_mean(prob * action, axis=-1))
            loss = - log_prob * reward

            # wc=prob*action
            #log  clip
            # print("prob",prob)
            # print("action",action)
            #rint("log_prob",log_prob)


            # loss = tf.reduce_mean(- log_prob * reward)








            # log_prob = tf.math.log(tf.reduce_mean(prob * action, axis=-1)) # (B, )
            # loss = - log_prob * reward


            # print("lr is ",self.learning_rate)
            # exit()
            # minimize = optimizer.minimize(loss)
            # print("loss is ",loss,"log_prob is ",log_prob,"reward is ",reward)
            grads = tape.gradient(loss, self.trainable_variables)

            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        if stateful:
            self.h = self.next_h
            self.c = self.next_c
            return loss
        else:

            return loss, self.next_h, self.next_c

    def sampling_word(self, prob):
        '''
        # Arguments:
            prob: numpy array, dtype=float, shape = (B, V),
        # Returns:
            action: numpy array, dtype=int, shape = (B, )
        '''
        action = np.zeros((self.batch_size,), dtype=np.int32)

        for i in range(self.batch_size):


            # g=prob[i]

            self.pro_nptype = np.array(prob[i])
            p=tf.reduce_sum(prob[i])
            # self.pro_nptype=prob[i].eval()
            # p = prob[i]
            # sum=tf.reduce_sum(p)
            # p=p/sum
            # print(prob[i].shape)

            # p /= p.sum()
            # print("p sum is ")
            # print(p.sum())

            # if p.sum()==float('nan'):
            #     for i in range(p):
            #         print (i)

            # print(p.shape)
            # print(type(p))

                    # exit()
            #add
            action[i] = np.random.choice(self.vocabrary_size, p=self.pro_nptype)
        return action

    def sampling_sentence(self, T, BOS=1):
        '''
        # Arguments:
            T: int, max time steps
        # Optional Arguments:
            BOS: int, id for Begin Of Sentence
        # Returns:
            actions: numpy array, dtype=int, shape = (B, T)
        '''
        self.reset_rnn_state()
        action = np.zeros([self.batch_size, 1], dtype=np.int32)
        action[:, 0] = BOS
        actions = action

        for _ in range(T):
            # print("_in T",_)
            prob = self.predict(action)
            # for i in range(self.batch_size):
            #
            #     p = prob[i]
            #
            #     for k in prob[i]:
            #         # print("k is " ,k)
            #         if np.isnan(k):
            #             print("batch size", self.batch_size)
            #
            #             print("when k is nan")
            #             print(prob[i])
            #             print(p)
            action = self.sampling_word(prob).reshape(-1, 1)

            actions = np.concatenate([actions, action], axis=-1)
        # Remove BOS
        actions = actions[:, 1:]
        self.reset_rnn_state()
        return actions
    # def sampling_sentence(self, T, BOS=1):
    #     '''
    #     # Arguments:
    #         T: int, max time steps
    #     # Optional Arguments:
    #         BOS: int, id for Begin Of Sentence
    #     # Returns:
    #         actions: numpy array, dtype=int, shape = (B, T)
    #     '''
    #
    #     self.reset_rnn_state()
    #     action = np.zeros([self.batch_size, 1], dtype=np.int32)
    #     action[:, 0] = BOS
    #     actions = action
    #
    #     for _ in range(T):
    #         prob = self.predict(action)
    #         # print("samping sentence ,the type of prob and shape")
    #         # print(type(prob))
    #         # print(prob.shape)
    #         # print(prob)
    #         # print("end samping print")
    #
    #         action = self.sampling_word(prob).reshape(-1, 1)
    #
    #         actions = np.concatenate([actions, action], axis=-1)
    #     # Remove BOS
    #     actions = actions[:, 1:]
    #     self.reset_rnn_state()
    #     return actions
    def predict(self, state, stateful=True):
        '''
        Predict next action(word) probability
        # Arguments:
            state: np.array, previous word ids, shape = (B, 1)
        # Optional Arguments:
            stateful: bool, default is True
                if True, update rnn_state(h, c) to Generator.h, Generator.c
                    and return prob.
                else, return prob, next_h, next_c without updating states.
        # Returns:
            prob: np.array, shape=(B, V)
        '''
        # state = state.reshape(-1, 1)
        # feed_dict = {
        #     self.state_in : state,
        #     self.h_in : self.h,
        #     self.c_in : self.c}
        # prob, next_h, next_c =
        #
        #     self.sess.run(
        #     [self.prob, self.next_h, self.next_c],
        #     feed_dict)

        # print("here is before call")
        # print(self.h.shape)

        prob=self.__call__(self.h,self.c,state)

        if stateful:
            self.h = self.next_h
            self.c = self.next_c
            return prob
        else:
            return prob, self.next_h, self.next_c



    def generate_samples(self, T, g_data, num, output_file):
        '''
        Generate sample sentences to output file
        # Arguments:
            T: int, max time steps
            g_data: SeqGAN.utils.GeneratorPretrainingGenerator
            num: int, number of sentences
            output_file: str, path
        '''
        sentences = []
        print("mp1")
        number_epoch=num // self.batch_size + 1
        for _ in range(number_epoch):
            d=num // self.batch_size + 1
            actions = self.sampling_sentence(T)
            actions_list = actions.tolist()
            print("\r number epoch"+str(number_epoch)+"g_samles ".format(_) + str(_), end="")

            for sentence_id in actions_list:
                sentence = [g_data.id2word[action] for action in sentence_id if action != 0 and action != 2]
                sentences.append(sentence)

        output_str = ''

        for i in range(num):
            output_str += ' '.join(sentences[i]) + '\n'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_str)

    def save(self, path):
        weights = []
        for layer in self.layers:
            w = layer.get_weights()
            weights.append(w)
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load(self, path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)

# class Discriminator(keras.Model):
#     def __init__(self,vocabrary_size, emb_dim, hidden_dim,dropout=0.1):
#         super(Discriminator, self).__init__()
#         self.vocabrary_size =vocabrary_size
#         self.emb_dim=emb_dim
#         self.hidden_dim=hidden_dim
#
#         self.layer1=keras.layers.Embedding(self.vocabrary_size, self.emb_dim, mask_zero=True, name='Embedding')
#         self.layer2=keras.layers.LSTM(hidden_dim)
#
#
#         self.layer4=keras.layers.Dropout(dropout,name='Dropout')
#         self.layer5=keras.layers.Dense(1,activation='sigmoid',name='FC')


    # def __call__(self,input,is_training=False):
    #     x=self.layer1(input)
    #     x=self.layer2(x)
    #     x=self.Highway(x,num_layers=1)
    #     x=self.layer4(x)
    #     x=self.layer5(x)
    #     return x
def Discriminator(V, E, H=64, dropout=0.1):

    input = keras.Input(shape=(None,), dtype='int32', name='Input')   # (B, T)
    out = keras.layers.Embedding(V, E, mask_zero=True, name='Embedding')(input)  # (B, T, E)
    out = keras.layers.LSTM(H,activation="sigmoid")(out)#if don't use sigmoid here ,code will raise    skip optimization
    out = Highway(out, num_layers=1)
    out = keras.layers.Dropout(dropout, name='Dropout')(out)
    out = keras.layers.Dense(1, activation='sigmoid', name='FC')(out)

    discriminator = keras.Model(input, out)
    return discriminator

def VariousConv1D(x, filter_sizes, num_filters, name_prefix=''):
    '''
    Layer wrapper function for various filter sizes Conv1Ds
    # Arguments:
        x: tensor, shape = (B, T, E)
        filter_sizes: list of int, list of each Conv1D filter sizes
        num_filters: list of int, list of each Conv1D num of filters
        name_prefix: str, layer name prefix
    # Returns:
        out: tensor, shape = (B, sum(num_filters))
    '''
    conv_outputs = []
    for filter_size, n_filter in zip(filter_sizes, num_filters):
        conv_name = '{}VariousConv1D/Conv1D/filter_size_{}'.format(name_prefix, filter_size)
        pooling_name = '{}VariousConv1D/MaxPooling/filter_size_{}'.format(name_prefix, filter_size)
        conv_out = keras.Conv1D(n_filter, filter_size, name=conv_name)(x)  # (B, time_steps, n_filter)
        conv_out = keras.GlobalMaxPooling1D(name=pooling_name)(conv_out)  # (B, n_filter)
        conv_outputs.append(conv_out)
    concatenate_name = '{}VariousConv1D/Concatenate'.format(name_prefix)
    out = keras.Concatenate(name=concatenate_name)(conv_outputs)
    return out

def Highway(x, num_layers=1, activation='relu', name_prefix=''):
    '''
    Layer wrapper function for Highway network
    # Arguments:
        x: tensor, shape = (B, input_size)
    # Optional Arguments:
        num_layers: int, dafault is 1, the number of Highway network layers
        activation: keras activation, default is 'relu'
        name_prefix: str, default is '', layer name prefix
    # Returns:
        out: tensor, shape = (B, input_size)
    '''
    input_size = keras.backend.int_shape(x)[1]
    for i in range(num_layers):
        gate_ratio_name = '{}Highway/Gate_ratio_{}'.format(name_prefix, i)
        fc_name = '{}Highway/FC_{}'.format(name_prefix, i)
        gate_name = '{}Highway/Gate_{}'.format(name_prefix, i)

        gate_ratio = keras.layers.Dense(input_size, activation='sigmoid', name=gate_ratio_name)(x)
        fc = keras.layers.Dense(input_size, activation=activation, name=fc_name)(x)
        x = keras.layers.Lambda(lambda args: args[0] * args[2] + args[1] * (1 - args[2]), name=gate_name)([fc, x, gate_ratio])
    return x
