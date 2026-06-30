import socket
import cv2  
import numpy as np

WIDTH = 1280
HEIGHT = 1280

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("0.0.0.0", 7007)) #alternative to 0.0.0.0 is 127.0.0.1
s.listen(1)

print("Waiting for Quest connection...")
comm_socket, addr = s.accept()
print("Connected:", addr)

def recvall(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

while True:
    print("waiting for 16-byte header...")
    header = recvall(comm_socket, 16)

    if header is None:
        print("header is None / connection closed")
        break

    print("got header bytes:", len(header), header)

    width = int.from_bytes(header[0:4], "big")
    height = int.from_bytes(header[4:8], "big")
    fmt = int.from_bytes(header[8:12], "big")
    size = int.from_bytes(header[12:16], "big")

    print("parsed header:", width, height, fmt, size)

    if width <= 0 or height <= 0 or size <= 0 or size > 50_000_000:
        print("invalid header, stopping")
        break

    print("waiting for payload:", size)
    data = recvall(comm_socket, size)

    if data is None:
        print("payload is None / connection closed")
        break

    print("got payload:", len(data))

    arr = np.frombuffer(data, dtype=np.uint8)

    if fmt == 2:
        img = arr.reshape((height, width))
    elif fmt == 5:
        rgba = arr.reshape((height, width, 4))
        img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    else:
        print("unsupported format:", fmt)
        continue

    print("showing frame")
    cv2.imshow("Quest Feed", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

comm_socket.close()
s.close()
cv2.destroyAllWindows()