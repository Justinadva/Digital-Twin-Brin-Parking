import cv2
import pickle
import os
import numpy as np

# Folder dan path penyimpanan
save_folder = 'A_Mobil_Positioning_Advanced'
os.makedirs(save_folder, exist_ok=True)

save_path = os.path.join(save_folder, 'mobil_positioning_parallelogram.pkl')
image_save_path = os.path.join(save_folder, 'layout_parking_spaces_parallelogram.png')

# Muat data jika sudah ada
try:
    with open(save_path, 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

current_polygon = []

# Fungsi interaksi mouse
def mouseClick(events, x, y, flags, params):
    global current_polygon, posList
    if events == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
        if len(current_polygon) == 4:
            posList.append((len(posList) + 1, current_polygon))
            current_polygon = []
            with open(save_path, 'wb') as f:
                pickle.dump(posList, f)
    elif events == cv2.EVENT_RBUTTONDOWN:
        for i, (id, points) in enumerate(posList):
            contour = np.array(points)
            if cv2.pointPolygonTest(contour.astype(np.float32), (x, y), False) >= 0:
                posList.pop(i)
                break
        with open(save_path, 'wb') as f:
            pickle.dump(posList, f)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Setup Slot Parkir Miring", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Setup Slot Parkir Miring", 1280, 720)
cv2.setMouseCallback("Setup Slot Parkir Miring", mouseClick)

while True:
    success, img = cap.read()
    if not success:
        print("❌ Gagal membuka kamera.")
        break

    # Gambar slot yang sudah disimpan
    for id, points in posList:
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cx = sum([p[0] for p in points]) // 4
        cy = sum([p[1] for p in points]) // 4
        cv2.putText(img, str(id), (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Gambar titik sementara
    for point in current_polygon:
        cv2.circle(img, point, 5, (0, 0, 255), -1)

    cv2.imshow("Setup Slot Parkir Miring", img)
    key = cv2.waitKey(1)
    if key == 27:
        cv2.imwrite(image_save_path, img)
        break

cap.release()
cv2.destroyAllWindows()
