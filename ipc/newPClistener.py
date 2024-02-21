import rpyc
import time


if __name__ == "__main__":
    print("Listener Connected")
    c = rpyc.connect("localhost", 18861)
    while(True):
        now = c.root.get_time()
        print(now)
        #s = time.asctime(now)
        #print(s)
        time.sleep(3)
