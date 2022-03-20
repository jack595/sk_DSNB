# -*- coding:utf-8 -*-
# @Time: 2022/3/7 22:17
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: CommonVariables.py
import numpy as np

class CommonVariables:
    map_tag = {"AfterPulse":0, "pES":1, "eES":2}
    tag2str = {item:key for key,item in map_tag.items()}

