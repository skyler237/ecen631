
class Buffer:
    def __init__(self, buffer_size):
        self.size = buffer_size
        self.buffer = []

    def push(self, data):
        if hasattr(data, "copy"):
            self.buffer.insert(0, data.copy())
        else:
            self.buffer.insert(0, data.copy())
        while len(self.buffer) > self.size:
            self.buffer.pop()

    def fill(self, frame):
        self.clear()
        for i in range(0,self.size):
            self.add_frame(frame)

    def pop(self):
        return self.buffer.pop()

    def peek(self, i):
        if i < self.size:
            return self.buffer[i]
        else:
            print("Invalid index: {0}".format(i))

    def last(self):
        return self.peek(-1)

    def set_size(self, size):
        self.size = size

    def get_all(self):
        return self.buffer

    def cnt(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []

class FrameBuffer(Buffer):

    def add_frame(self, frame):
        self.push(frame)

    def get_frames(self):
        return self.get_all()