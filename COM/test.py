# from threading import Thread
# from time import sleep, ctime
import serial
import serial.tools.list_ports

plist = list(serial.tools.list_ports.comports())
print(len(plist))

if len(plist) <= 0:
    print("The Serial port can't find!")
else:
    plist_0 = list(plist[0])
    plist_1 = list(plist[1])
    serialName1 = plist_0[0]
    serialName2 = plist_1[0]
    print(serialName1)
    # print(serialName2)
    serialFd1 = serial.Serial(serialName1, 9600, timeout=0.5)
    serialFd2 = serial.Serial(serialName2, 9600, timeout=0.5)
    # print("check which port was really used", serialFd.name)
    # while 1:
    #     str = input("请输入接收到的数据:")
    #     print((str+'\n').encode())
    #     writeBit = serialFd.write((str+'\n').encode())
    #     print(writeBit)
    #     print(serialFd.read_all())
    # serialFd.close()
    # str = input("请输入接收到的数据:")
while 1:
    writeBit = serialFd1.write(("test"+'\n').encode())
    print(serialFd2.readline())  
    # serialFd1.close()
    # serialFd2.close()

# def func(name, sec):
#     print('---开始---', name, '时间', ctime())
#     sleep(sec)
#     print('***结束***', name, '时间', ctime())


# t1 = Thread(target=func, args=('第一个线程', 1))
# t2 = Thread(target=func, args=('第二个线程', 2))

# t1.start()
# t2.start()

# t1.join()
# t2.join()
