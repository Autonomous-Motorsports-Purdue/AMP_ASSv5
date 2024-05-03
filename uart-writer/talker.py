import rpyc
import serial
import time

START_BYTE = 254
END_BYTE = 255

class Talker(rpyc.Service): #rpyc.Service allows me to import the service into the class
    def __init__(self):
        # configure the serial connections (the parameters differs on the device you are connecting to)
        self.ser = serial.Serial(
            port='/dev/ttyUSB0',
            baudrate=115200
        )

        self.curr_v = 0
        self.curr_s = 0

    def on_connect(self, conn): #keyword method that allows for diagnostics on connect
        print("Talker connected")
        return "Talked connected"
    def on_disconnect(self, conn): #keyword like above, but disconnect
        print("Talker disconnected")
        return "Talker disconnected"
    def exposed_update_velocity(self, new_v): #the exposed keyword at the front allows the object to be accesible. 
        # shifting values into UART accepted range (128-255) (zero at 191)
        if (new_v < 0):
            new_v = 0
        elif(new_v > 255):
            new_v = 255
        new_v =  new_v >> 1
        
        self.curr_v = new_v
    def exposed_update_steering(self, new_s): #the exposed keyword at the front allows the object to be accesible. 
        # shifting values into UART accepted range (128-255) (zero at 191)
        if(new_s <= -63):
            new_s = 0
        elif(new_s >= 64):
            new_s = 127
        else:
            new_s = new_s + 64
        
        self.curr_s = new_s
    def exposed_get_time(self): #the exposed keyword at the front allows the object to be accesible. 
        return time.asctime(time.localtime())
    def exposed_write_serial(self): #the exposed keyword at the front allows the object to be accesible. 
        
        # send start byte
        self.ser.write(START_BYTE.to_bytes(1, "little"))
        
        
        # send current velocity and steering
        self.ser.write(self.curr_v.to_bytes(1, "little"))
        self.ser.write(self.curr_s.to_bytes(1, "little"))
        
        # send end byte
        self.ser.write(END_BYTE.to_bytes(1, "little"))

if __name__ == "__main__": 
    from rpyc.utils.server import ThreadedServer #threaded server to run everything
    t = ThreadedServer(Talker, port=18864) #service and port declareation. 
    t.start()
