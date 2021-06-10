# -*- coding:utf-8 -*-
# @Time: 2021/5/21 16:19
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: PlotIntoPDF.py
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
def PlotIntoPDF(v_figs:list, name_out_pdf):
    with PdfPages(name_out_pdf) as pdf:
        for fig in v_figs:
            pdf.savefig(fig)
