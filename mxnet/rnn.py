import mxnet as mx


def rnn_step(input_, state, wx, wh, b, num_hidden):
    # num_hidden = wh.shape[0]
    # update
    i2h = mx.sym.FullyConnected(data=input_,
                                weight=wx,
                                no_bias=True,
                                num_hidden=num_hidden,
                                name='i2h')
    # memory
    h2h = mx.sym.FullyConnected(data=state,
                                weight=wh,
                                bias=b,
                                num_hidden=num_hidden,
                                name='h2h')
    # activation
    return mx.sym.Activation(data=i2h + h2h, act_type="tanh")
    # return hidden


def rnn_unroll(wx, wh, b, wa, ba,
               seq_len, n_inputs, num_hidden, n_labels, batch_size):

    # n_embed = n_inputs
    # we = mx.sym.Variable('we')
    hidden = mx.sym.Variable('init_h')

    data = mx.sym.Variable('data')
    # data = mx.sym.transpose(data=data, axis=[0, 1])
    label = mx.sym.Variable('label')
    x_mat = mx.sym.SliceChannel(
        data=data, num_outputs=seq_len, axis=1, squeeze_axis=False)

    output = []
    label = mx.sym.SliceChannel(
        data=label, num_outputs=seq_len, axis=1, squeeze_axis=True)

    # label = mx.sym.transpose(label, axis=[1, 0, 2])
    for i in xrange(seq_len):
        hidden = rnn_step(input_=x_mat[i],
                          state=hidden,
                          wx=wx,
                          wh=wh,
                          b=b,
                          num_hidden=num_hidden)
        # states.append(hidden)
        fc = mx.sym.FullyConnected(data=hidden, weight=wa, bias=ba,
                                   num_hidden=n_labels,
                                   name='pred',
                                   )
        sm = mx.sym.LogisticRegressionOutput(data=fc, label=label[i],
                                             name='logistic')
        output.append(sm)

    output = mx.sym.Concat(*output, dim=1)  # important to set dim=1
    return mx.sym.Reshape(output, shape=(batch_size, seq_len, 1))


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class DataIter(mx.io.DataIter):
    def __init__(self, n_samples, batch_size, seq_len, n_inputs,
                 num_label, init_states, create_dataset):
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.n_samples = n_samples

        self.seq_len = seq_len
        self.num_label = num_label
        self.init_states = init_states
        # self.init_state_names = [x[0] for x in self.init_states]
        # self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [
            ('data', (batch_size, seq_len, n_inputs))] + init_states
        self.provide_label = [('label', (batch_size, seq_len, num_label))]

        X, T = create_dataset(n_samples, seq_len)
        self.data = X
        self.labels = T

    def __iter__(self):
        batch_data = []
        batch_label = []
        # batch_label_weight = []
        for i in xrange(self.n_samples):

            batch_data.append(self.data[i])
            batch_label.append(self.labels[i])
            # batch_label_weight.append(label_weight)
            if len(batch_data) == self.batch_size:
                # + self.init_state_arrays
                data_all = [mx.nd.array(batch_data), ]
                label_all = [mx.nd.array(batch_label), ]

                data_names = ['data']  # + self.init_state_names
                label_names = ['label']
                batch_data = []
                batch_label = []
                yield SimpleBatch(data_names, data_all, label_names, label_all)

    # def reset(self):
    #     pass
