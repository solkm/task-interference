#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:27:23 2024

@author: Sol
"""

from __future__ import division
from __future__ import print_function

from psychrnn.tasks.task import Task
from psychrnn.backend.models.basic import Basic
from psychrnn.backend.simulation import BasicSimulator
import tensorflow as tf
import numpy as np
from abc import ABCMeta
ABC = ABCMeta('ABC', (object,), {})
from time import time
from os import makedirs, path

class RNN_SH2(Basic):
    def train(self, task, train_params):
        """ 
        Arguments:
            task (:class:`~psychrnn.tasks.task.Task` object): the task to train on. task.:func:`batch_generator` () will be called to get the generator for the task to train on.
            train_params (dict, optional): Dictionary of training parameters containing the following possible keys:

                :Dictionary Keys:
                    * **learning_rate** (*float, optional*) -- Sets learning rate if use default optimizer Default: .001
                    * **training_iters** (*int, optional*) -- Number of iterations to train for Default: 50000.
                    * **loss_epoch** (*int, optional*) -- Compute and record loss every 'loss_epoch' epochs. Default: 10.
                    * **verbosity** (*bool, optional*) -- If true, prints information as training progresses. Default: True.
                    * **save_weights_path** (*str, optional*) -- Where to save the model after training. Default: None
                    * **save_training_weights_epoch** (*int, optional*) -- Save training weights every 'save_training_weights_epoch' epochs. Weights only actually saved if :data:`training_weights_path` is set. Default: 100.
                    * **training_weights_path** (*str, optional*) -- What directory to save training weights into as training progresses. Default: None.
                    * **curriculum** (`~psychrnn.backend.curriculum.Curriculum` *object, optional*) -- not implemented here
                    * **optimizer** (`tf.compat.v1.train.Optimizer <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Optimizer>`_ *object, optional*) -- What optimizer to use to compute gradients. Default: `tf.train.AdamOptimizer <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer>`_ (learning_rate=:data:`train_params`['learning_rate']` ).
                    * **clip_grads** (*bool, optional*) -- If true, clip gradients by norm 1. Default: True
                    * **fixed_weights** (*dict, optional*) -- By default all weights are allowed to train unless :data:`fixed_weights` or :data:`W_rec_train`, :data:`W_in_train`, or :data:`W_out_train` are set. Default: None. Dictionary of weights to fix (not allow to train) with the following optional keys:

                        Fixed Weights Dictionary Keys (in case of :class:`~psychrnn.backend.models.basic.Basic` and :class:`~psychrnn.backend.models.basic.BasicScan` implementations)
                            * **W_in** (*ndarray(dtype=bool, shape=(:attr:`N_rec`. :attr:`N_in` *)), optional*) -- True for input weights that should be fixed during training.
                            * **W_rec** (*ndarray(dtype=bool, shape=(:attr:`N_rec`, :attr:`N_rec` *)), optional*) -- True for recurrent weights that should be fixed during training.
                            * **W_out** (*ndarray(dtype=bool, shape=(:attr:`N_out`, :attr:`N_rec` *)), optional*) -- True for output weights that should be fixed during training.

                        :Note:
                            In general, any key in the dictionary output by :func:`get_weights` can have a key in the fixed_weights matrix, however fixed_weights will only meaningfully apply to trainable matrices.

                    * **performance_cutoff** (*float*) -- not implemented here
                    * **performance_measure** (*function*) -- not implemented here

                        :Arguments:
                            * **trial_batch** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Task stimuli for :attr:`N_batch` trials.
                            * **trial_y** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Target output for the network on :attr:`N_batch` trials given the :data:`trial_batch`.
                            * **output_mask** (*ndarray(dtype=bool, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Output mask for :attr:`N_batch` trials. True when the network should aim to match the target output, False when the target output can be ignored.
                            * **output** (*ndarray(dtype=bool, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Output to compute the accuracy of. ``output`` as returned by :func:`psychrnn.backend.rnn.RNN.test`.
                            * **epoch** (*int*): Current training epoch (e.g. perhaps the performance_measure is calculated differently early on vs late in training)
                            * **losses** (*list of float*): List of losses from the beginning of training until the current epoch.
                            * **verbosity** (*bool*): Passed in from :data:`train_params`.

                        :Returns:
                            *float*

                            Performance, greater when the performance is better.
        Returns:
            tuple:
            * **losses** (*list of float*) -- List of losses, computed every :data:`loss_epoch` epochs during training.
            * **training_time** (*float*) -- Time spent training.
            * **initialization_time** (*float*) -- Time spent initializing the network and preparing to train.

        """
        if not self.is_built:
            self.build()

        t0 = time()
        # --------------------------------------------------
        # Extract params
        # --------------------------------------------------
        learning_rate = train_params.get('learning_rate', .001)
        training_iters = train_params.get('training_iters', 50000)
        loss_epoch = train_params.get('loss_epoch', 10)
        verbosity = train_params.get('verbosity', True)
        save_weights_path = train_params.get('save_weights_path', None)
        save_training_weights_epoch = train_params.get('save_training_weights_epoch', 100)
        training_weights_path = train_params.get('training_weights_path', None)
        optimizer = train_params.get('optimizer', tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate))
        clip_grads = train_params.get('clip_grads', True)
        fixed_weights = train_params.get('fixed_weights', None)  # array of zeroes and ones. One indicates to pin and not train that weight.
 
        # --------------------------------------------------
        # Make weights folder if it doesn't already exist.
        # --------------------------------------------------
        if save_weights_path != None:
            if path.dirname(save_weights_path) != "" and not path.exists(path.dirname(save_weights_path)):
                makedirs(path.dirname(save_weights_path))

        # --------------------------------------------------
        # Make train weights folder if it doesn't already exist.
        # --------------------------------------------------
        if training_weights_path != None:
            if path.dirname(training_weights_path) != "" and not path.exists(path.dirname(training_weights_path)):
                makedirs(path.dirname(training_weights_path))

        # --------------------------------------------------
        # Compute gradients
        # --------------------------------------------------
        grads = optimizer.compute_gradients(self.reg_loss)

        # --------------------------------------------------
        # Fixed Weights
        # --------------------------------------------------
        if fixed_weights is not None:
            for i in range(len(grads)):
                (grad, var) = grads[i]
                name = var.name[len(self.name) + 1:-2]
                if name in fixed_weights.keys():
                    grad = tf.multiply(grad, (1 - fixed_weights[name]))
                    grads[i] = (grad, var)

        # --------------------------------------------------
        # Clip gradients
        # --------------------------------------------------
        if clip_grads:
            grads = [(tf.clip_by_norm(grad, 1.0), var)
                     if grad is not None else (grad, var)
                     for grad, var in grads]

        # --------------------------------------------------
        # Call the optimizer and initialize variables
        # --------------------------------------------------
        optimize = optimizer.apply_gradients(grads)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.is_initialized = True

        # --------------------------------------------------
        # Record training time for performance benchmarks
        # --------------------------------------------------
        t1 = time()

        # --------------------------------------------------
        # Training loop
        # --------------------------------------------------
        epoch = 1
        losses = []

        batch_size = task.get_task_params()['N_batch']
        n_back = task.get_task_params()['K'] - 1
        hist = dict()
        hist['choice'] = np.zeros((batch_size, n_back))
        hist['correct'] = np.zeros((batch_size, n_back))
        hist['dsl'] = np.zeros((batch_size, n_back))
        hist['dsf'] = np.zeros((batch_size, n_back))
        hist['task'] = np.zeros((batch_size, n_back))
        hist['t_ind'] = np.zeros((batch_size, 0))
        hist['t_sess'] = np.zeros((batch_size, 1))
        
        while (epoch-1) * batch_size < training_iters:
            batch_x, batch_y, output_mask, hist = task.train_batch_generator(hist)
            self.sess.run(optimize, feed_dict={self.x: batch_x, self.y: batch_y, self.output_mask: output_mask})
            output, state_var = self.test(batch_x)
            choice = np.argmax(output[:, -1, 2:6], axis=1) + 1
            hist['choice'][:, -1] = choice
            # --------------------------------------------------
            # Output batch loss
            # --------------------------------------------------
            if epoch % loss_epoch == 0:
                reg_loss = self.sess.run(self.reg_loss, feed_dict={self.x: batch_x, self.y: batch_y, self.output_mask: output_mask})
                losses.append(reg_loss)
                if verbosity:
                    print("Iter " + str(epoch * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(reg_loss))
                    print('accuracy: ', 1 - np.count_nonzero(choice - hist['correct'][:, -1])/choice.shape[0])
            # --------------------------------------------------
            # Save intermediary weights
            # --------------------------------------------------
            if epoch % save_training_weights_epoch == 0:
                if training_weights_path is not None:
                    self.save(training_weights_path + str(epoch))
                    if verbosity:
                        print("Training weights saved in file: %s" % training_weights_path + str(epoch))
                        
            epoch += 1
            
        t2 = time()
        if verbosity:
            print("Optimization finished!")

        # --------------------------------------------------
        # Save final weights
        # --------------------------------------------------
        if save_weights_path is not None:
            self.save(save_weights_path)
            if verbosity:
                print("Model saved in file: %s" % save_weights_path)

        # --------------------------------------------------
        # Return losses, training time, initialization time
        # --------------------------------------------------
        return hist, losses, (t2 - t1), (t1 - t0)

    def test(self, trial_batch):
        """ Test the network on a certain task input.

        Arguments:
            trial_batch ((*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Task stimulus to run the network on. Stimulus from :func:`psychrnn.tasks.task.Task.get_trial_batch`, or from next(:func:`psychrnn.tasks.task.Task.batch_generator` ).
        
        Returns:
            tuple:
            * **outputs** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*) -- Output time series of the network for each trial in the batch.
            * **states** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_rec` *))*) -- Activity of recurrent units during each trial.
        """
        if not self.is_built:
            self.build()

        if not self.is_initialized:
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.is_initialized = True

        # --------------------------------------------------
        # Run the forward pass on trial_batch
        # --------------------------------------------------
        outputs, states = self.sess.run([self.predictions, self.states],
                                        feed_dict={self.x: trial_batch})

        return outputs, states

class Task_SH2(Task):
    def __init__(self, dt=10, tau=100, T=1800, N_batch=100, K=10, mem_noise=0.5, 
                 vis_noise=0.5, dat=None, dat_inds=None, targCorrect=True, testall=False, 
                 fixedDelay=None, fixedSL=[None, None], fixedSF=[None, None], 
                 selftest_weights_path=None, selftest_network_params=None):
        N_in = 6*(K-1)+3
        N_out = 7
        super().__init__(N_in, N_out, dt, tau, T, N_batch)
        self.K = K
        self.mem_noise = mem_noise
        self.vis_noise = vis_noise
        self.fixedDelay = fixedDelay
        self.fixedSL = fixedSL
        self.fixedSF = fixedSF
        self.fixedDelay = fixedDelay
        self.dat = dat
        self.dat_inds = dat_inds
        self.targCorrect = targCorrect
        self.testall = testall
        if testall==True:
            if (self.N_batch != dat_inds.shape[0]):
                print('N_batch does not match data shape')
        self.selftest_weights_path = selftest_weights_path
        self.selftest_network_params = selftest_network_params
        
    def train_batch_generator(self, hist):
        """ Generates train trials.

        Returns:
            tuple:

            * **stimulus** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Task stimuli for :attr:`N_batch` trials.
            * **target_output** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Target output for the network on :attr:`N_batch` trials given the :data:`stimulus`.
            * **output_mask** (*ndarray(dtype=bool, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Output mask for :attr:`N_batch` trials. True when the network should aim to match the target output, False when the target output can be ignored.
            * **trial_params** (*ndarray(dtype=dict, shape =(*:attr:`N_batch` *,))*): Array of dictionaries containing the trial parameters produced by :func:`generate_trial_params` for each trial in :attr:`N_batch`.

        """
        x_data = []
        y_data = []
        mask = []
        K = self.K
        dat = self.dat
        dat_inds = self.dat_inds
        N_batch = self.N_batch
        assert np.all(dat.loc[dat_inds, f'K{K}trainable']==1), f'dat_inds must be K{K} trainable'
        start_inds = dat_inds[dat.loc[dat_inds-1, f'K{K}trainable']==0] - (K-1)
        dat_inds_sorted = np.sort(dat_inds)
        end_inds = dat_inds_sorted[:-1][dat.loc[dat_inds_sorted[:-1]+1, f'K{K}trainable']==0]
        end_inds = np.append(end_inds, dat_inds_sorted[-1])
            
        hist['choice'] = np.append(hist['choice'], np.zeros((N_batch, 1)), axis=1)
        hist['correct'] = np.append(hist['correct'], np.zeros((N_batch, 1)), axis=1)
        hist['dsl'] = np.append(hist['dsl'], np.zeros((N_batch, 1)), axis=1)
        hist['dsf'] = np.append(hist['dsf'], np.zeros((N_batch, 1)), axis=1)
        hist['task'] = np.append(hist['task'], np.zeros((N_batch, 1)), axis=1)
        hist['t_ind'] = np.append(hist['t_ind'], np.zeros((N_batch, 1)), axis=1)
        hist['t_sess'] = np.append(hist['t_sess'], np.zeros((N_batch, 1)), axis=1)
        
        for trial in range(self.N_batch):
            if hist['t_sess'][trial, -2] == 0: # if it's the first epoch
                t_ind = np.random.choice(start_inds)
                hist['t_ind'][trial, -1] = t_ind
                hist['t_sess'][trial, -1] = 1
                
            elif np.isin(hist['t_ind'][trial, -2], end_inds) == False: # if the previous trial WAS NOT an end_ind
                t_ind = hist['t_ind'][trial, -2] + 1
                hist['t_ind'][trial, -1] = t_ind
                hist['t_sess'][trial, -1] = hist['t_sess'][trial, -2] + 1
            
            else: # if the previous trial WAS an end_ind, choose a new start ind and reset t_sess
                t_ind = np.random.choice(start_inds)
                hist['t_ind'][trial, -1] = t_ind
                hist['t_sess'][trial, -1] = 1
                
            hist['correct'][trial, -1] = dat.loc[t_ind, 'correct']
            hist['task'][trial, -1] = dat.loc[t_ind, 'task']
            hist['dsl'][trial, -1] = dat.loc[t_ind, 'dsl']/2
            hist['dsf'][trial, -1] = dat.loc[t_ind, 'dsf']/2
            
            params = dict()
            params['choice'] = np.array(hist['choice'][trial, -K:].copy(), dtype='int')
            params['correct'] = np.array(hist['correct'][trial, -K:].copy(), dtype='int')
            params['dsl'] = hist['dsl'][trial, -K:].copy()
            params['dsf']= hist['dsf'][trial, -K:].copy()
            params['task'] = hist['task'][trial, -K:].copy()
            
            if hist['t_sess'][trial, -1] < K:
                t_sess = int(hist['t_sess'][trial, -1])
                params['choice'] = np.zeros(K)
                params['correct'][:-t_sess] = 0
                params['dsl'][:-t_sess] = 0
                params['dsf'][:-t_sess] = 0
                params['task'][:-t_sess] = 0

            params['sf1'] = np.random.uniform(2,3)
            params['sf2'] = params['sf1'] + params['dsf'][-1]
            params['sl1'] = np.random.uniform(2,3)
            params['sl2'] = params['sl1'] + params['dsl'][-1]
            params['delay2_dur'] = np.random.uniform(300,500) # variable delay period
            
            if self.targCorrect==False:
                params['mk_choice'] = dat.loc[t_ind, 'choice']
                params['mk_task'] = dat.loc[t_ind, 'm_task']
            
            x, y, m = self.generate_trial(params)
            x_data.append(x)
            y_data.append(y)
            mask.append(m)
            
        return np.array(x_data), np.array(y_data), np.array(mask), hist
    
    def generate_selftest_params(self, i, choice_hist):
        """ see function below
        """
        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
            
        dat = self.dat
        K = self.K

        params = dict()
        params['trial_ind'] = i
        params['choice'] = choice_hist[-K:]
        params['correct'] = np.array(dat['correct'][i-K+1:i+1], dtype='int')
        params['dsf'] = np.array(dat['dsf'][i-K+1:i+1])/2
        params['dsl'] = np.array(dat['dsl'][i-K+1:i+1])/2
        params['task'] = np.array(dat['task'][i-K+1:i+1], dtype='int')
        params['m_task'] = np.array(dat['m_task'][i-K+1:i+1], dtype='int')
        
        dsf = params['dsf'][-1]
        params['sf1'] = np.random.uniform(2,3)
        params['sf2'] = params['sf1'] + dsf
        
        dsl = params['dsl'][-1]
        params['sl1'] = np.random.uniform(2,3)
        params['sl2'] = params['sl1'] + dsl
    
        if self.fixedSF[0] is not None:
            params['sf1'] = self.fixedSF[0]
            
            if self.fixedSF[1] is not None:
                params['sf2'] = self.fixedSF[1]
                dsf = self.fixedSF[1] - self.fixedSF[0]
            else:
                dsf = np.random.uniform(-1,1)
                params['sf2'] = self.fixedSF[0] + dsf
        
        else:
            assert self.fixedSF[1] is None, 'cannot fix only the second stimulus'
                
        if self.fixedSL[0] is not None:
            params['sl1'] = self.fixedSL[0]
            
            if self.fixedSL[1] is not None:
                params['sl2'] = self.fixedSL[1]
                dsl = self.fixedSL[1] - self.fixedSL[0]
            else:
                dsl = np.random.uniform(-1,1)
                params['sl2'] = self.fixedSL[0] + dsl
        
        else:
            assert self.fixedSL[1] is None, 'cannot fix only the second stimulus'
        
        if self.fixedDelay is not None:
            params['delay2_dur'] = self.fixedDelay
        else:
            params['delay2_dur'] = np.random.uniform(300, 500) # variable delay period
        
        return params
    
    def selftest_batch_generator(self):
        """ Generates test trials using MODEL (fully trained) trial history. 
        Initializes with monkey history K-1 trials back, and then computes model choice history up to the current trial.
        Runs slowly, only use for small numbers of trials. Otherwise see the sequential version.
        """
        
        x_data = []
        y_data = []
        mask = []
        params = []

        simulator = BasicSimulator(weights_path=self.selftest_weights_path, params=self.selftest_network_params)  
        dat = self.dat
        dat_inds = self.dat_inds
        K = self.K
    
        for trial in range(self.N_batch):
            
            # initialize with monkey history trials:
            if self.testall==False:
                j = np.random.choice(dat_inds)
            else:
                j = dat_inds[trial]
            i = j-(K-1)
            choice_hist = np.append(np.array(dat['choice'][i-K+1:i], dtype='int'), 0)
            p = self.generate_selftest_params(i, choice_hist)
            
            # iterate to current trial, adding model's choices to history:
            for n in range(K):
                test_inputs = np.zeros((1, self.N_steps, self.N_in))
                test_inputs[:, :, :], _, _ = self.generate_trial(p)
                output, _ = simulator.run_trials(test_inputs)
                model_choice = np.argmax(output[0, -1, 2:6]) + 1
                choice_hist[-1] = model_choice
                choice_hist = np.append(choice_hist, 0)
     
                p = self.generate_selftest_params(i+n, choice_hist)
            
            x,y,m = self.generate_trial(p)
            x_data.append(x)
            y_data.append(y)
            mask.append(m)
            params.append(p)

        return np.array(x_data), np.array(y_data), np.array(mask), np.array(params)
    
    def selftest_batch_generator_sequential(self):
        x_data = []
        y_data = []
        mask = []
        params = []

        simulator = BasicSimulator(weights_path=self.selftest_weights_path, params=self.selftest_network_params)  
        dat = self.dat
        dat_inds = self.dat_inds
        K = self.K
        assert np.array_equal(dat_inds, np.sort(dat_inds)), 'trials must be in order'
        
        i = dat_inds[0]
        choice_hist = np.append(np.array(dat['choice'][i-K+1:i], dtype='int'), 0)
        
        for trial in range(self.N_batch):
            i = dat_inds[trial]
            if trial==0 or dat_inds[trial] != dat_inds[trial-1]+1:
                choice_hist = np.append(np.array(dat['choice'][i-K+1:i], dtype='int'), 0)
                print(f'monkey hist init, trial {trial}')
            p = self.generate_selftest_params(i, choice_hist)
            x,y,m = self.generate_trial(p)
            x_data.append(x)
            y_data.append(y)
            mask.append(m)
            params.append(p)
            test_inputs = np.zeros((1, self.N_steps, self.N_in))
            test_inputs[:,:,:] = x
            output, _ = simulator.run_trials(test_inputs)
            model_choice = np.argmax(output[0, -1, 2:6]) + 1
            choice_hist[-1] = model_choice
            choice_hist = np.append(choice_hist, 0)

        return np.array(x_data), np.array(y_data), np.array(mask), np.array(params)
    
    def generate_trial_params(self, batch, trial):
        """"Define parameters for each trial using MONKEY trial history. Used in batch_generator.

        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.

        Returns:
            dict: Dictionary of trial parameters.
        """
        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
            
        dat = self.dat
        dat_inds = self.dat_inds
        K = self.K
        
        if self.testall==False:
            i = np.random.choice(dat_inds)
        else:
            i = dat_inds[trial]

        params = dict()
        params['trial_ind'] = i
        params['choice'] = np.array(dat['choice'][i-K+1:i+1], dtype='int')
        params['correct'] = np.array(dat['correct'][i-K+1:i+1], dtype='int')
        params['dsf'] = np.array(dat['dsf'][i-K+1:i+1])/2
        params['dsl'] = np.array(dat['dsl'][i-K+1:i+1])/2
        params['task'] = np.array(dat['task'][i-K+1:i+1], dtype='int')
        params['m_task'] = np.array(dat['m_task'][i-K+1:i+1], dtype='int')
        
        dsf = params['dsf'][-1]
        params['sf1'] = np.random.uniform(2,3)
        params['sf2'] = params['sf1'] + dsf
        
        dsl = params['dsl'][-1]
        params['sl1'] = np.random.uniform(2,3)
        params['sl2'] = params['sl1'] + dsl
    
        if self.fixedSF[0] is not None:
            params['sf1'] = self.fixedSF[0]
            
            if self.fixedSF[1] is not None:
                params['sf2'] = self.fixedSF[1]
                dsf = self.fixedSF[1] - self.fixedSF[0]
            else:
                dsf = np.random.uniform(-1,1)
                params['sf2'] = self.fixedSF[0] + dsf
        
        else:
            assert self.fixedSF[1] is None, 'cannot fix only the second stimulus'
                
        if self.fixedSL[0] is not None:
            params['sl1'] = self.fixedSL[0]
            
            if self.fixedSL[1] is not None:
                params['sl2'] = self.fixedSL[1]
                dsl = self.fixedSL[1] - self.fixedSL[0]
            else:
                dsl = np.random.uniform(-1,1)
                params['sl2'] = self.fixedSL[0] + dsl
        
        else:
            assert self.fixedSL[1] is None, 'cannot fix only the second stimulus'
        
        if self.fixedDelay is not None:
            params['delay2_dur'] = self.fixedDelay
        else:
            params['delay2_dur'] = np.random.uniform(300, 500) # variable delay period
        
        return params
    
    def batch_generator(self):
        """ Returns a generator for this task. Unchanged code, used in get_trial_batch().

        Returns:
            Generator[tuple, None, None]:

        Yields:
            tuple:

            * **stimulus** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Task stimuli for :attr:`N_batch` trials.
            * **target_output** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Target output for the network on :attr:`N_batch` trials given the :data:`stimulus`.
            * **output_mask** (*ndarray(dtype=bool, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Output mask for :attr:`N_batch` trials. True when the network should aim to match the target output, False when the target output can be ignored.
            * **trial_params** (*ndarray(dtype=dict, shape =(*:attr:`N_batch` *,))*): Array of dictionaries containing the trial parameters produced by :func:`generate_trial_params` for each trial in :attr:`N_batch`.
        
        """

        batch = 1
        while batch > 0:

            x_data = []
            y_data = []
            mask = []
            params = []
            # ----------------------------------
            # Loop over trials in batch
            # ----------------------------------
            for trial in range(self.N_batch):
                # ---------------------------------------
                # Generate each trial based on its params
                # ---------------------------------------
                p = self.generate_trial_params(batch, trial)
                x,y,m = self.generate_trial(p)
                x_data.append(x)
                y_data.append(y)
                mask.append(m)
                params.append(p)

            batch += 1

            yield np.array(x_data), np.array(y_data), np.array(mask), np.array(params)
            
    def get_trial_batch(self):
        """Get a batch of trials.
    
        Wrapper for :code:`next(self._batch_generator)`.
    
        Returns:
            tuple:
    
            * **stimulus** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_in` *))*): Task stimuli for :attr:`N_batch` trials.
            * **target_output** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Target output for the network on :attr:`N_batch` trials given the :data:`stimulus`.
            * **output_mask** (*ndarray(dtype=bool, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Output mask for :attr:`N_batch` trials. True when the network should aim to match the target output, False when the target output can be ignored.
            * **trial_params** (*ndarray(dtype=dict, shape =(*:attr:`N_batch` *,))*): Array of dictionaries containing the trial parameters produced by :func:`generate_trial_params` for each trial in :attr:`N_batch`.
    
        """
        return next(self._batch_generator)
    
    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.
    
        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.
    
        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()
    
        Returns:
            tuple:
    
            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.
    
        """
        
        delay1_dur = 500
        stim_dur = 200
        delay2_dur = params['delay2_dur']
        delay3_dur = 150
        choice_dur = 750 - delay2_dur
        
        if delay1_dur + 2*stim_dur + delay2_dur + delay3_dur + choice_dur != self.T:
            print('error: trial event times do not add up!')
        
        stim1_on = delay1_dur
        stim2_on = stim1_on + stim_dur + delay2_dur
        go = stim2_on + stim_dur + delay3_dur
        K = self.K
        # ----------------------------------
        # Initialize with input noise
        # ----------------------------------
        vis_noise = self.vis_noise
        mem_noise = self.mem_noise
        fix_noise = 0.2
        
        x_t = np.zeros(self.N_in)
        x_t[-1] += np.sqrt(2 * fix_noise**2) * np.random.standard_normal(1)
        x_t[-3:-1] += np.sqrt(2 * vis_noise**2) * np.random.standard_normal(2)
        x_t[:-3] += np.sqrt(2 * mem_noise**2) * np.random.standard_normal(self.N_in-3)
        
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)
    
        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        dsl = params['dsl']
        dsf = params['dsf']
        choice = params['choice']
        correct = params['correct']
        sf1 = params['sf1']
        sf2 = params['sf2']
        sl1 = params['sl1']
        sl2 = params['sl2']
        task = params['task']

        dStim = np.zeros(K-1)
        for j in range(K-1):
            if params['choice'][j] > 2.5:
                dStim[j] = dsl[j]
            else:
                dStim[j] = dsf[j]
        
        # ----------------------------------
        # Compute values
        # ----------------------------------
        
        for i in range(0, K-1): # constant trial history inputs for the duration of the trial
            if choice[i] != 0:
                x_t[6*i] += 0.2 + dStim[i] # stimulus change amount, based on the chosen task (signals perceptual difficulty)
                x_t[6*i + choice[i]] += 1 # one-hot encoded choice
                if choice[i] == correct[i]:
                    x_t[6*i+5] += 1 # reward
                else:
                    x_t[6*i+5] += -1 # error
                
        if time > 100:
            if self.targCorrect:
                y_t[int(task[-1]-1)] += 1 # report the task
            else:
                y_t[int(params['mk_task']-1)] += 1 # report the monkey's chosen task
                
            mask_t[:2] *= 1.5

        if stim1_on < time < stim1_on + stim_dur: # stim1 input
            x_t[-3] += sf1
            x_t[-2] += sl1
        
        if stim2_on < time < stim2_on + stim_dur: # stim2 input
            x_t[-3] += sf2
            x_t[-2] += sl2        
    
        if time >= go:
            if self.targCorrect:
                y_t[int(correct[-1]+1)] = 1
            else:
                y_t[int(params['mk_choice']+1)] = 1
                
            mask_t[2:6] *= 5 # choice unit outputs should be weighed heavily after "go"
        else:
            x_t[-1] += 1 # fixation period
            y_t[-1] = 1
        
        return x_t, y_t, mask_t
