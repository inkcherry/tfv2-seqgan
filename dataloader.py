import numpy as np
class Data_Loader():
    def __init__(self,batch_size,filename):
        self.batch_size=batch_size
        self.all_tokens = []

        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    self.all_tokens.append(parse_line)

        self.num_batch = int(len(self.all_tokens) / self.batch_size)
        self.all_tokens = self.all_tokens[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.all_tokens), self.num_batch, 0)
        self.cur_index = 0

    def next_batch(self):
        batch_data=self.sequence_batch[self.cur_index]
        self.cur_index=(self.cur_index + 1) % self.num_batch
        return batch_data


data_loader = Data_Loader(64,"real_data.txt")


t=data_loader.next_batch()
print(t.shape)