from threading import Thread
import time
def count():
    i = 0
    while True:
        print("current time{}".format(i))
        time.sleep(0.5)
        i+=1
def work1():
    for i in range(10):
        print("Work thread 1...")
        time.sleep(1)
def work2():
    for i in range(10):
        print("Work thread 2...")
        time.sleep(0.25)

if __name__ == "__main__":
    thread1 = Thread(target=work1)
    thread2 = Thread(target=work2)
    thread1.start()
    thread2.start()