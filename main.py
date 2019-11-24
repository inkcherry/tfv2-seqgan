import numpy as np
from train import Trainer
from get_config import get_config
import tensorflow as tf
config = get_config('config.ini')
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#
physical_devices = tf.config.experimental.list_physical_devices('GPU')
# for i in physical_devices:
    # tf.config.experimental.set_virtual_device_configuration(
    #     i,
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])


#
# # tf.config.gpu.set_per_process_memory_fraction(0.75)


for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)

print(config["g_lr"])
trainer = Trainer(config["batch_size"],
                config["max_length"],
                config["g_e"],
                config["g_h"],
                config["d_e"],
                config["d_h"],
                config["d_dropout"],
                path_pos=config["path_pos"],
                path_neg=config["path_neg"],
                g_lr=config["g_lr"],
                d_lr=config["d_lr"],
                n_sample=config["n_sample"],
                generate_samples=config["generate_samples"])




# Pretraining for adversarial training 对抗训练的pretrain
trainer.pre_train(g_epochs=config["g_pre_epochs"],
                d_epochs=config["d_pre_epochs"],
                g_pre_path=config["g_pre_weights_path"],
                d_pre_path=config["d_pre_weights_path"],
                g_lr=config["g_pre_lr"],
                d_lr=config["d_pre_lr"])









trainer.load_pre_train(config["g_pre_weights_path"], config["d_pre_weights_path"])
trainer.reflect_pre_train()

trainer.train(steps=10,
            g_steps=1,
            head=10,
            g_weights_path=config["g_weights_path"],
            d_weights_path=config["d_weights_path"])

trainer.save(config["g_weights_path"], config["d_weights_path"])


trainer.load(config["g_weights_path"], config["d_weights_path"])

trainer.test()

# trainer.generate_txt(config["g_test_path"], config["generate_samples"])