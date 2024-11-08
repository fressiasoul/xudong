import multiprocessing
import os

def run_file(filename):
    os.system('python {}'.format(filename))

if __name__ == '__main__':
    # 创建两个进程，分别运行两个Python文件

    # p1 = multiprocessing.Process(target=run_file, args=('DQN.py',))
    p2 = multiprocessing.Process(target=run_file, args=('DQNbs.py',))
    p3 = multiprocessing.Process(target=run_file, args=('DQNen.py',))


    # 启动两个进程
    # p1.start()
    #p2.start()
    p3.start()


    # 等待两个进程结束
    # p1.join()
    #p2.join()
    p3.join()

