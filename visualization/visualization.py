

import matplotlib.pyplot as plt
import sys, os, shutil




def save(iters, iters_acc):
         
        plt.figure(figsize=(10, 4))
        plt.ylabel('accuracy', fontsize=12)
        plt.xlabel('iteration', fontsize=12)
        plt.grid(True)
        plt.plot(iters, iters_acc, color='0.4')
        plt.savefig('./train_acc', dpi=1200)
   




