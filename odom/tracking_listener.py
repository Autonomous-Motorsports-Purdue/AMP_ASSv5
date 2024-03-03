import rpyc
import time

if __name__ == "__main__":
    c = rpyc.connect("localhost", 90001)
    print("Listener Connected")
    while True:
        pos = c.root.get_position()
        print(pos)
        #s = time.asctime(now)
        #print(s)
        time.sleep(0.1)
