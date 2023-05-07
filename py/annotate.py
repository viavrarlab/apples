import os
import json
import itertools


import cv2 as cv


DIR = "data/apples.v2i.coco/valid"
PATHS = "data/paths.txt"
ANNOTATIONS = "data/_annotations.coco.json"


def on_click(event, x, y, _, userdata):
    if event == cv.EVENT_LBUTTONDOWN:
        cls = userdata[0]
        color = (0, 0, 255)
    elif event == cv.EVENT_RBUTTONDOWN:
        cls = userdata[1]
        color = (0, 255, 0)
    else:
        return
    if cls:
        cv.line(userdata[2][0], cls[-1], (x, y), color, 3)
        cv.imshow("bgr", userdata[2][0])
    cls.append((x, y))


userdata = ([], [], [None])
data = {}


def draw(cls, color):
    for xy, zw in zip(cls, cls[1:]):
        cv.line(userdata[2][0], xy, zw, color, 3)


def get_annotations(id, classes):
    annotations = []
    for cls, segments in enumerate(classes, 1):
        for segment in segments:
            min_x = min(x for x, _ in segment)
            max_x = max(x for x, _ in segment)
            min_y = min(y for _, y in segment)
            max_y = max(y for _, y in segment)
            x = max_x - min_x
            y = max_y - min_y
            annotations.append({
                "area": x * y,
                "bbox": [min_x, min_y, x, y],
                "category_id": cls,
                # "id": 0,
                "image_id": id,
                "iscrowd": 0,
                "segmentation": [list(itertools.chain.from_iterable(segment))],
            })
    return annotations


def save_annotations(data):
    with open(PATHS, "w") as file:
        file.write("\n".join(sorted(data)))
    data = sorted(data.items())
    with open(ANNOTATIONS, "w") as file:
        json.dump({
            "annotations": [
                annotation.__setitem__("id", id) or annotation
                for id, annotation in enumerate(itertools.chain.from_iterable(get_annotations(id, classes) for id, (_, (_, classes)) in enumerate(data)))
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "1",
                    "supercategory": "0",
                },
                {
                    "id": 2,
                    "name": "2",
                    "supercategory": "0",
                },
            ],
            "images": [{
                # "date_captured": "",
                "file_name": path,
                "height": height,
                "id": id,
                "license": 1,
                "width": width,
            } for id, (path, ((height, width, _), _)) in enumerate(data)],
            "info": {
                # "contributor": "",
                # "date_created": "",
                # "description": "",
                # "url": "",
                "version": "1",
                # "year": "",
            },
            "licenses": [{
                "id": 1,
                # "name": "",
                # "url": "",
            }],
        }, file, ensure_ascii=False, indent=4, sort_keys=True)


with open(PATHS, "r") as file:
    paths = {line.strip() for line in file.readlines()}


cv.namedWindow("bgr", cv.WINDOW_GUI_NORMAL)
cv.resizeWindow("bgr", 900, 900)
cv.setMouseCallback("bgr", on_click, userdata)
for root, _, files in os.walk(DIR):
    for file in sorted(files):
        path = os.path.join(root, file)
        print(path)
        if path in paths:
            continue
        try:
            bgr = cv.imread(path, cv.IMREAD_COLOR)
            userdata[2][0] = bgr.copy()
        except:
            continue
        cv.imshow("bgr", userdata[2][0])
        classes = ([], [])
        while True:
            while (key := cv.waitKey(100) & 0xff) < 0:
                continue
            if key == ord("w"):
                if userdata[0]:
                    classes[0].append(list(userdata[0]))
                    print(f"saved {userdata[0]}")
                    userdata[0].clear()
                elif userdata[1]:
                    classes[1].append(list(userdata[1]))
                    print(f"saved {userdata[1]}")
                    userdata[1].clear()
            elif key == ord("s"):
                if userdata[1]:
                    userdata[1].pop()
                elif userdata[0]:
                    userdata[0].pop()
                elif classes[1]:
                    classes[1].pop()
                elif classes[0]:
                    classes[0].pop()
                else:
                    continue
                userdata[2][0] = bgr.copy()
                for cls in classes[0]:
                    draw(cls, (0, 0, 255))
                draw(userdata[0], (0, 0, 255))
                for cls in classes[1]:
                    draw(cls, (0, 255, 0))
                draw(userdata[1], (0, 255, 0))
                cv.imshow("bgr", userdata[2][0])
            elif key == ord("q"):
                break
        data[path] = (bgr.shape, classes)
        save_annotations(data)
