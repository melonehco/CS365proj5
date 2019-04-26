"""
KerasTimer.py
A subclass of keras callbacks that we can use to take the times
being output to the console and write them to TrainingTimerResults
every time a training session is done for optimizeNetwork

Zena Abulhab and Melody Mao
CS365 Spring 2019
Project 5
"""
import keras
import time

class KerasTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)