#!/usr/bin/python3

import logging
import numpy as np
from research.numpy_socket_test.numpysocket.numpysocket import NumpySocket
import socket
import time

logger = logging.getLogger('simple client')
logger.setLevel(logging.INFO)

with NumpySocket() as s:
    s.connect(("localhost", 9999))
    
    logger.info("sending numpy array:")
    frame = np.random.random((10000000))
    print(time.time())
    s.sendall(frame)

