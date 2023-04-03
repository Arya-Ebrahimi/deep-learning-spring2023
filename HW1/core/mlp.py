import tensorflow  as tf
import numpy as np
import datetime

class NeuralNet(tf.keras.Model):
    def __init__(self, num_layers, hiddens, num_classes):
        super(NeuralNet, self).__init__()
        
        self.num_layers = num_layers
        self.hiddens = hiddens
        self.num_classes = num_classes
        
        self.fc = []
        for i in range(self.num_layers-1):
            self.fc.append(tf.keras.layers.Dense(self.hiddens[i], activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.RandomUniform()))
        self.out = tf.keras.layers.Dense(self.num_classes, kernel_initializer=tf.keras.initializers.RandomUniform())
        
        
    def call(self, x, is_training=False):
        for i in range(self.num_layers-1):
            x = self.fc[i](x)
        x = self.out(x)
        
        if not is_training:
            x = tf.nn.softmax(x)
        
        return x

class MLP():
    def __init__(self, config):
        self.learning_rate = config['learning_rate']
        self.num_classes = config['num_classes']
        self.num_features = config['num_features']
        self.training_steps = config['training_steps']
        self.batch_size = config['batch_size']
        self.n_layers = config['n_layers']
        self.hiddens = config['hiddens']
        self.save_ratio = 100
        self._load_data()
        self.model = NeuralNet(self.n_layers, self.hiddens, self.num_classes)
        optim = config['optimizer']
        if optim == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        elif optim == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(self.learning_rate)
        elif optim == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
            
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
        self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
        self.name = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.train_log_dir = 'logs/' + self.name + '/train'
        self.test_log_dir = 'logs/' + self.name + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        
    def _load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='/home/arya/Desktop/DL/HW1/dataset/mnist.npz')
        x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
        x_train, x_test = x_train.reshape([-1, self.num_features]), x_test.reshape([-1, self.num_features])
        x_train, x_test = x_train / 255., x_test / 255.
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.repeat().shuffle(5000).batch(self.batch_size).prefetch(1)
        self.train_data = train_data
        self.x_test = x_test
        self.y_test = y_test
        
    def _loss_fn(self, x, y):
        y = tf.cast(y, tf.int64)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
        return tf.reduce_mean(loss)

    def _accuracy(self, y_pred, y_true):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)
    
    
    def _optimize(self, x, y):
        with tf.GradientTape() as tape:
            pred = self.model(x, is_training=True)
            loss = self._loss_fn(pred, y)
            
            trainable_vars = self.model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars) 
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            self.train_loss(loss)
            self.train_accuracy(pred, y)
            
    def train(self):
        for step, (batch_x, batch_y) in enumerate(self.train_data.take(self.training_steps), 1):
            self._optimize(batch_x, batch_y)
            
            if step % self.save_ratio == 0:    
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step)
                    
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()
                
                pred = self.model(batch_x, is_training=True)
                loss = self._loss_fn(pred, batch_y)
                acc = self._accuracy(pred, batch_y)
                print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
                
        print('training completed')
        
        pred = self.model(self.x_test)
        print('Test accuracy: %f' % self._accuracy(pred, self.y_test))
    