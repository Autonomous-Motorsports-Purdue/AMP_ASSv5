import rpyc
import time

if __name__ == "__main__":
    c = rpyc.connect("localhost", 9001)
    print("Listener Connected")
    while True:
        pos = c.root.get_position()
        if pos is not None:
            print(pos)
        #s = time.asctime(now)
        #print(s)
        time.sleep(0.1)
