import cv2
import numpy as np

from sort import Sort

# SORT tracking algorithm initialization
sort_max_age = 50
sort_min_hits = 2
sort_iou_thresh = 0.2
sort_tracker = Sort(max_age=sort_max_age,
                    min_hits=sort_min_hits,
                    iou_threshold=sort_iou_thresh)

# Sorting algorithm
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    color = [int(p * (label ** 2 - label + 1) % 255) for p in palette]
    return tuple(color)


def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def draw_boxes(img, bbox, identities=None, categories=None,
               names=None, color_box=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
        label = str(id)

        if color_box:
            color = compute_color_for_labels(id)
        else:
            color = (255, 191, 0)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1 - 3 * h), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2 * h), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    [255, 255, 255], 1)
        cv2.circle(img, data, 3, color, -1)

    return img


def _display_detected_tracks(dets, img, color_box=None):
    dets_to_sort = np.empty((0, 6))

    for x1, y1, x2, y2, conf, detclass in dets.cpu().detach().numpy():
        dets_to_sort = np.vstack((dets_to_sort,
                                  np.array([x1, y1, x2, y2,
                                            conf, detclass])))

    # Run SORT
    tracked_dets = sort_tracker.update(dets_to_sort)

    # Draw boxes for visualization
    if len(tracked_dets) > 0:
        bbox_xyxy = tracked_dets[:, :4]
        identities = tracked_dets[:, 8]
        categories = tracked_dets[:, 4]
        names = None
        draw_boxes(img, bbox_xyxy, identities,
                   categories, names, color_box)

    return img