# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
import tensorflow as tf

def get_bounds(data, factor=10):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)
    return (min_x, max_x, min_y, max_y)

def slerp(p0, p1, t):
    """Spherical interpolation. 球形插值""" 
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1

def lerp(p0, p1, t):
    """Linear interpolation. 线性插值"""
    return (1.0 - t) * p0 + t * p1

def strokes_to_lines(strokes):
    """Convert stroke-3 format to polyline format.！！！！"""
    x = 0
    y = 0
    lines = []
    line = []
    for i in range(len(strokes)):
        if strokes[i, 2] == 1:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
            lines.append(line)
            line = []
        else:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
    return lines


def lines_to_strokes(lines):
    """Convert polyline format to stroke-3 format."""
    eos = 0
    strokes = [[0, 0, 0]]
    for line in lines:
        linelen = len(line)
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[i][0], line[i][1], eos])
    strokes = np.array(strokes)
    strokes[1:, 0:2] -= strokes[:-1, 0:2]
    return strokes[1:, :]


def augment_strokes(strokes, prob=0.0):
    """Perform data augmentation by randomly dropping out strokes.随机删除笔画执行数据增强"""
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = candidate
            prev_stroke = stroke
            result.append(stroke)
    return np.array(result)

def scale_bound(stroke, average_dimension=10.0):
    """Scale an entire image to be less than a certain size."""
    # stroke is a numpy array of [dx, dy, pstate], average_dimension is a float.
    # modifies stroke directly.
    bounds = get_bounds(stroke, 1)
    max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    stroke[:, 0:2] /= (max_dimension / average_dimension)

def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3] # !!!!
    return result


def clean_strokes(sample_strokes, factor=100):
    """Cut irrelevant end points, scale to pixel space and store as integer."""
    # Useful function for exporting data to .json format.
    copy_stroke = []
    added_final = False
    for j in range(len(sample_strokes)):
        finish_flag = int(sample_strokes[67][4]) #最后一个点的最后一位为1
        if finish_flag == 0:
            print('数据有问题呀，util文件119行')
        else:
            copy_stroke.append([0, 0, 1])
            added_final = True
            break
    if not added_final:
        copy_stroke.append([0, 0, 1])
    return copy_stroke


def to_big_strokes(stroke, max_len=68):
    """Converts from stroke-3 to stroke-5 format and pads to given length."""
    # (But does not insert special start token).

    result = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    result[0:l, :] = stroke[:, :]
    return result

def get_max_len(strokes):
    """Return the maximum length of an array of strokes."""
    max_len = 0
    for stroke in strokes:
        ml = len(stroke)
        if ml > max_len:
            max_len = ml
    return max_len


class DataLoader(object):
    """Class for loading data."""
    #tf.logging.info('utils-按batch大小取数据函数============================================')
    def __init__(self,
             source,indices,
             batch_size=64,
             max_seq_length=68,
             scale_factor=1.0,
             random_scale_factor=0.0,
             limit=1000):
        self.batch_size = batch_size  # minibatch size
        self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
        self.scale_factor = scale_factor  # divide offsets by this factor
        self.random_scale_factor = random_scale_factor  
        self.limit = limit
        self.start_stroke_token = [0.0, 0.0, 1.0, 0.0, 0.0] 
        self.preprocess(source)
        
    def preprocess(self, source):
        """Remove entries from strokes having > max_seq_length points."""
        self.source = []

        for i in range(len(source)):
            source_data = source[i]
            source_data = np.array(source_data, dtype=np.float32)
            source_data[:, 0:2] /= self.scale_factor
            self.source.append(source_data)
        self.num_batches = int(len(source) / self.batch_size)

    def random_sample(self):
        """Return a random sample, in stroke-3 format as used by draw_strokes."""
        sample_source = np.copy(random.choice(self.source))
        return sample_source

    def random_scale(self, data):
        """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
        x_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        result = np.copy(data)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result

    def calculate_normalizing_scale_factor(self):
        """Calculate the normalizing factor explained in appendix of sketch-rnn."""
        data = []
        for i in range(len(self.source)):
            for j in range(len(self.source[i])):
                data.append(self.source[i][j, 0])
                data.append(self.source[i][j, 1])
        data = np.array(data)
        return np.std(data)
    
    def normalize(self, scale_factor=None):
        if scale_factor is None:
            scale_factor = self.calculate_normalizing_scale_factor()
        self.scale_factor = scale_factor
        for i in range(len(self.source)):
            self.source[i][:, 0:2] /= self.scale_factor
                        
    def _get_batch_from_indices(self, indices):
        """Given a list of indices, return the potentially augmented batch."""
        source_batch = []
        seq_len = []
        for idx in range(len(indices)):
            i = indices[idx]
            data = self.random_scale(self.source[i])
            data_copy = np.copy(data)
            source_batch.append(data_copy)
            length = len(data_copy)
            seq_len.append(length)
        seq_len = np.array(seq_len, dtype=int)
        return source_batch, self.pad_batch(source_batch, self.max_seq_length), seq_len

    def pad_batch(self, batch, max_len):
        """Pad the batch to be stroke-5 bigger format as described in paper."""
        result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
        assert len(batch) == self.batch_size
        for i in range(self.batch_size):
            l = len(batch[i])
            assert l <= max_len
            result[i, 0:l, :] = batch[i][:, :]
            result[i, 1:, :] = result[i, :-1, :]
            result[i, 0, :] = 0.0
        return result
