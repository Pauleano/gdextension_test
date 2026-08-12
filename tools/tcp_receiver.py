import socket
import cv2  
import numpy as np
import os

save_dir = "tools/images"
os.makedirs(save_dir, exist_ok=True)

WIDTH =640
HEIGHT =480

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


def recv_markers(sock):
    """Marker block appended after the image payload by main_3d.gd::_send_frame_tcp:
    count u32, then per marker id u32 + 8 big-endian f32 = the 4 corners as x0,y0,..,x3,y3.

    These are the corners the DEVICE's detector found, already mapped back out of its
    downscaled detection frame, so they line up with the full-size image in the payload.
    Returns (ids, corners) ready for cv2.aruco.drawDetectedMarkers, or None on a closed
    connection. MUST be read for every frame, including ones we cannot display -- skipping
    it leaves the unread bytes in the stream and desyncs every frame after it.
    """
    raw = recvall(sock, 4)
    if raw is None:
        return None
    count = int.from_bytes(raw, "big")
    if count > 1000:
        print("implausible marker count, stream desynced:", count)
        return None

    ids, corners = [], []
    for _ in range(count):
        record = recvall(sock, 4 + 8 * 4)
        if record is None:
            return None
        ids.append(int.from_bytes(record[0:4], "big"))
        pts = np.frombuffer(record, dtype=">f4", count=8, offset=4).astype(np.float32)
        # drawDetectedMarkers wants one (1, 4, 2) array per marker
        corners.append(pts.reshape(1, 4, 2))
    return ids, corners

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

    # Before any `continue` below: the marker block belongs to this frame and has to leave the
    # stream either way.
    markers = recv_markers(comm_socket)

    if markers is None:
        print("marker block is None / connection closed")
        break

    ids, corners = markers

    arr = np.frombuffer(data, dtype=np.uint8)

    # Always end up 3-channel: drawDetectedMarkers needs colour to draw its green borders and red
    # first-corner dot, and frombuffer/reshape alone would hand it a read-only array to draw into.
    if fmt == 2:
        img = cv2.cvtColor(arr.reshape((height, width)), cv2.COLOR_GRAY2BGR)
    elif fmt == 5:
        rgba = arr.reshape((height, width, 4))
        img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    else:
        print("unsupported format:", fmt)
        continue

    # Drawn into a COPY, so `img` stays the clean frame: the 's' key below writes into
    # tools/images, which is the input set for cameraCalibration.py -- a burnt-in overlay there
    # would corrupt the corner detection it runs.
    view = img.copy()

    if ids:
        print("drawing markers:", ids)
        cv2.aruco.drawDetectedMarkers(view, corners, np.array(ids, dtype=np.int32).reshape(-1, 1))

    print("showing frame")
    cv2.imshow("Quest Feed", view)

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