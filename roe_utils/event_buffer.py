import numpy as np
# from vizdoom import *
import math


# import mysql.connector

class EventBuffer:

    def __init__(self, n, capacity, event_clip=0.01):
        self.n = n
        self.capacity = capacity
        self.idx = 0
        self.events = []
        self.event_clip = event_clip

    def record_events(self, events, frame):
        if len(self.events) < self.capacity:
            self.events.append(events)
        else:
            self.events[self.idx] = events
            if self.idx + 1 < self.capacity:
                self.idx += 1
            else:
                self.idx = 0

    def intrinsic_reward(self, events, vector=False):
        if len(self.events) == 0:
            if vector:
                return np.ones(self.n)
            return events[0]

        mean = np.mean(self.events, axis=0)
        clip = np.clip(mean, self.event_clip, np.max(mean))

        div = np.divide(np.ones([clip.size]), clip)
        # div = np.divide(np.ones([mean.size]), mean)

        mul = np.multiply(div, events)
        clip2 = np.clip(mul, a_min=None, a_max=2)

        if vector:
            return mul
        return np.sum(clip2)

    def get_event_mean(self):
        if len(self.events) == 0:
            return np.zeros(self.n)
        mean = np.mean(self.events, axis=0)
        return mean

    def get_event_rewards(self):
        return self.intrinsic_reward(np.ones(self.n), vector=True)

