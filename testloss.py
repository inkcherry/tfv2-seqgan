import numpy   as np
from model import  Generator
import os
import  tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')

for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)

class TestGenerator():
    # def sub_test(self, actual, expected, msg=None):
    #     with self.subTest(actual=actual, expected=expected):
    #         self.assertEqual(actual, expected, msg=msg)


    def test_generator(self):
        B = 4
        E = 2
        H = 3
        V = 5
        generator = Generator(B, V, E, H)
        BOS = 1
        x = [BOS] * B
        x = np.array(x).reshape(B, 1)

        prob = generator.predict(x)

        # self.sub_test(prob.shape, (B, V), msg='output shape test')
        # self.assertAlmostEqual(B, np.sum(prob), places=1, msg='softmax test')

        for i in range(100):
            prob2 = generator.predict(x)

        generator.reset_rnn_state()
        prob3 = generator.predict(x)

        # self.assertNotAlmostEqual(prob[0, 0], prob2[0, 0], places=10, msg='stateful test')
        # self.assertAlmostEqual(prob[0, 0], prob3[0, 0], places=7, msg='stateful test')

        print("x is ",x)
        action = np.array([1, 2, 3, 4]).reshape(4,1)

        reward = np.array([0.1, 0, 0.1, 0.8]).reshape(4,1)

        loss = generator.update(x, action, reward)
        for i in range(500):
            generator.reset_rnn_state()
            loss = generator.update(x, action, reward)

            if i % 100 == 0:
                print("loss is", loss)
                generator.reset_rnn_state()
                prob = generator.predict(x)
                print(prob[0])
        # self.sub_test(np.argmax(prob[0]), 4, 'RL optimization test')

t=TestGenerator()
t.test_generator()