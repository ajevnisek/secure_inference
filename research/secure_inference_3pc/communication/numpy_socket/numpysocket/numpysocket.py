#!/usr/bin/env python3

import socket
import logging
import numpy as np
from io import BytesIO


class NumpySocket(socket.socket):
    def sendall(self, frame):
        if not isinstance(frame, np.ndarray):
            raise TypeError("input frame is not a valid numpy array") # should this just call super intead?

        out = self.__pack_frame(frame)
        super().sendall(out)
        logging.debug("frame sent")


    def recv(self, bufsize=4096):

        data = super().recv(bufsize)

        if len(data) == 0:
            return np.array([])
        else:
            frameBuffer = bytearray()
            length_str, ignored, data = data.partition(b':')
            length = int(length_str)

            frameBuffer += data

            while len(frameBuffer) < length:
                data = super().recv(bufsize)
                frameBuffer += data

        frame = np.load(BytesIO(frameBuffer), allow_pickle=True)['frame']
        return frame

    def accept(self):
        fd, addr = super()._accept()
        sock = NumpySocket(super().family, super().type, super().proto, fileno=fd)
        
        if socket.getdefaulttimeout() is None and super().gettimeout():
            sock.setblocking(True)
        return sock, addr
    

    @staticmethod
    def __pack_frame(frame):
        f = BytesIO()
        np.savez(f, frame=frame)
        
        packet_size = len(f.getvalue())
        header = '{0}:'.format(packet_size)
        header = bytes(header.encode())  # prepend length of array

        out = bytearray()
        out += header

        f.seek(0)
        out += f.read()
        return out
