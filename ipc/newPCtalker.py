import rpyc
import time

class Talker(rpyc.Service): #rpyc.Service allows me to import the service into the class
    def on_connect(self, conn): #keyword method that allows for diagnostics on connect
        print("Talker connected")
        return "Talked connected"
    def on_disconnect(self, conn): #keyword like above, but disconnect
        print("Talker disconnected")
        return "Talker disconnected"
    def exposed_get_time(self): #the exposed keyword at the front allows the object to be accesible. 
        return time.asctime(time.localtime())

if __name__ == "__main__": 
    from rpyc.utils.server import ThreadedServer #threaded server to run everything
    t = ThreadedServer(Talker, port=18861) #service and port declareation. 
    t.start()
