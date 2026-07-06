import socket
import cv2  
import numpy as np
import os

save_dir = "tools/images"
os.makedirs(save_dir, exist_ok=True)

WIDTH = 1280
HEIGHT = 1280

#have to create a tcp connection on socket 7007
#adb reverse tcp:7007 tcp:7007
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

image_num=0

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

    k = cv2.waitKey(100) & 0xFF

    if k  == ord('q'):
        break
    elif k== ord('s'):

        filename = os.path.join(save_dir, f"img{image_num}.jpg")
        success = cv2.imwrite(filename, img)

        if success:
            print(f"image saved: {filename}")
            image_num += 1
        else:
            print(f"could not save image: {filename}")

comm_socket.close()
s.close()
cv2.destroyAllWindows()