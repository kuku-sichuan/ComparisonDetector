# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from collections import OrderedDict
from configs.config import Config

net_config = Config()

if net_config.DATASET_NAME == 'tct':
    NAME_LABEL_MAP = OrderedDict({
        "back_ground": 0,
        'ascus': 1,
        'asch': 2,
        'lsil': 3,
        'hsil': 4,
        'scc': 5,
        'agc': 6,
        'trichomonas': 7,
        'candida': 8,
        'flora': 9,
        'herps': 10,
        'actinomyces': 11,

    })

else:
    assert 'please set label dict!'

def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

LABEl_NAME_MAP = get_label_name_map()