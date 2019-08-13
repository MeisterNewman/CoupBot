import os, select
import pickle

class CommunicationChannel:  #A communication channel sends one object at a time, blocking until it is recieved. It can send objects bigger than the pipe limit.
    def __init__(self):
        self.signal_out, self.signal_in = os.pipe()
        self.info_out, self.info_in = os.pipe()
        self.poller = select.poll()
        self.poller.register(self.info_out, select.POLLIN)
    def write(self, obj):
        dump=pickle.dumps(obj, -1)
        os.write(self.signal_in, (" " + str(len(dump))).encode())
        os.write(self.info_in, dump)

    def has_data(self):
        return (self.poller.poll(0)!=[])

    def idle_until_data(self):
        self.poller.poll()

    def read(self):
        lengths = os.read(self.signal_out, 2 ** 16).decode()[1:]
        if " " in lengths:
            os.write(self.signal_in, lengths[lengths.index(" "):].encode())
            length = lengths[:lengths.index(" ")]
        else:
            length = lengths
        length=int(length)
        data = b''
        while len(data)<length:
            data+=os.read(self.info_out, length-len(data))
        return pickle.loads(data)



