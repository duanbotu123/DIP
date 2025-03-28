import abc
import torch.optim
from .timer import Timer
from .logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from ..main_utils import cfg

from ..osx_model import get_model

# # dynamic dataset import
# for i in range(len(cfg.trainset_3d)):
#     exec('from ' + cfg.trainset_3d[i] + ' import ' + cfg.trainset_3d[i])
# for i in range(len(cfg.trainset_2d)):
#     exec('from ' + cfg.trainset_2d[i] + ' import ' + cfg.trainset_2d[i])
# exec('from ' + cfg.testset + ' import ' + cfg.testset)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Demoer(Base):
    def __init__(self, test_epoch=None):
        if test_epoch is not None:
            self.test_epoch = int(test_epoch)
        super(Demoer, self).__init__()

    def _make_model(self):
        print('Load checkpoint from {}'.format(cfg.pretrained_model_path))

        # prepare network
        print("Creating graph...")
        model = get_model()
        model = DataParallel(model).cuda()
        ckpt = torch.load(cfg.pretrained_model_path)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in ckpt['network'].items():
            k = k.replace('module.backbone', 'module.encoder').replace('body_rotation_net', 'body_regressor').replace(
                'hand_rotation_net', 'hand_regressor')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()

        self.model = model