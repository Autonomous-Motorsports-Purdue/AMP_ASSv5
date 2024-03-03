import rpyc
import time

if __name__ == "__main__":
    c = rpyc.connect("localhost", 9001)
    print("Listener Connected")
    while True:
        if pos:
            pos = c.root.get_position()
        print(pos.get())
        #s = time.asctime(now)
        #print(s)
        time.sleep(0.1)
