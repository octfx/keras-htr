from project.gui.gui import HtrGui
from project.htr.models import limit_gpu_memory

if __name__ == '__main__':
    limit_gpu_memory(1024)
    gui = HtrGui(model_path='model')
