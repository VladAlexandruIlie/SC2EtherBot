import numpy as np
# from vizdoom import *
import math


# import mysql.connector

class EventBuffer:

    def __init__(self, n, capacity, event_clip=0.01, events_number=12):
        self.n = n
        self.events_number = events_number
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
        eventsCopy = np.copy(events)
        if len(self.events) == 0:
            if vector:
                return np.ones(self.n)
            clip_temp = np.clip(events[0], -100, 100)
            sum0 = np.sum(clip_temp)
            return sum0

        mean = np.mean(self.events, axis=0)
        clip = np.clip(mean, self.event_clip, np.max(mean))

        div = np.divide(np.ones([clip.size]), clip)

        mul = np.multiply(div, eventsCopy)

        clip2 = np.clip(mul, a_min=-100, a_max=100)

        if vector:
            return mul
        return np.sum(clip2)

    def get_event_mean(self):
        if len(self.events) == 0:
            return np.zeros(self.events_number)

        mean = np.mean(self.events, axis=0)
        return mean

    def get_event_rewards(self):
        return self.intrinsic_reward(np.ones(self.n), vector=True)

    def get_events_number(self):
        return self.events_number

    def set_event_number(self, event_capacity):
        self.events_number = event_capacity
