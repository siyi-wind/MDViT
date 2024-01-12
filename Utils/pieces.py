import collections
import torch
import numpy as np

class DotDict(dict):
    '''
    enable to use dot to search the dict
    dict = {'name':cici}
    dotdict = DotDict(dict)
    dotdict.name
    '''
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        v = DotDict(v)
                    if isinstance(v, list):
                        self.__convert(v)
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    v = DotDict(v)
                elif isinstance(v, list):
                    self.__convert(v)
                self[k] = v

    def __convert(self, v):
        for elem in range(0, len(v)):
            if isinstance(v[elem], dict):
                v[elem] = DotDict(v[elem])
            elif isinstance(v[elem], list):
                self.__convert(v[elem])

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]


def load_pretrain(model, pre_s_dict):
    ''' Load state_dict in pre_model to model
    Solve the problem that model and pre_model have some different keys'''
    s_dict = model.state_dict()
    # remove fc weights and bias
    # use new dict to store states, record missing keys
    missing_keys = []
    new_state_dict = collections.OrderedDict()
    for key in s_dict.keys():
        if key in pre_s_dict.keys():
            new_state_dict[key] = pre_s_dict[key]
        else:
            new_state_dict[key] = s_dict[key]
            missing_keys.append(key)
    print('{} keys are not in the pretrain model:'.format(len(missing_keys)), missing_keys)
    # load new s_dict
    model.load_state_dict(new_state_dict)
    return model


class AvgMeter(object):
    '''
    from TransFuse 
    https://github.com/Rayicer/TransFuse/blob/main/utils/utils.py
    '''
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def dice_per_img(score, target):
    '''calculate dice loss for each image in a batch
    score and target are numpy array, output is a numpy array  B'''
    
    target = np.atleast_2d(target.astype(bool))
    B = target.shape[0]
    target = target.reshape((B,-1))
    score = np.atleast_2d(score.astype(bool)).reshape((B,-1))

    intersection = np.count_nonzero(target & score, axis=1)

    size_i1 = np.count_nonzero(target, axis=1).astype(np.float32)
    size_i2 = np.count_nonzero(score, axis=1).astype(np.float32)
    
    try:
        dc = 2. * intersection / (size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    return dc


    target = target.float().view(B,-1)
    score = score.view(B,-1)
    smooth = 1e-7
    intersect = torch.sum(score*target,dim=[1])
    y_sum = torch.sum(target * target,dim=[1])
    z_sum = torch.sum(score * score,dim=[1])
    dice = (2 * intersect+smooth) / (z_sum + y_sum+smooth)
    # loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    # loss = 1 - loss
    return dice

    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    
    intersection = numpy.count_nonzero(result & reference)
    
    size_i1 = numpy.count_nonzero(result)
    size_i2 = numpy.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

if __name__ == '__main__':
    # a = {'name':'cici', 'd': {'status':True, 'personality': 'kind'}}
    # a = DotDict(a)
    # print(a.d.personality)

    a = np.array([[1.,0.,1.], [1.,1.,0.]])
    b = np.array([[True,False,True], [True,False,False]])
    c = dice_per_img(a,b)
    import medpy.metric.binary as metrics
    d = metrics.dc(a,b)
    print(c)
    print(d)

    # test = np.array([[True,False,True]])
    # print(np.count_nonzero(test,axis=1))