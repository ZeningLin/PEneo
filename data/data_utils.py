import random
from typing import Dict, List, Tuple, Union

import numpy as np


def box_two_point_convert(box: Union[List[float], Dict[str, float]]):
    if isinstance(box, List) and len(box) == 4:
        return box

    assert len(box) == 8, f"Box should be List or Dict that contains 4 or 8 values."

    x_set, y_set = set(), set()
    if isinstance(box, List):
        for i, bv in enumerate(box):
            if i % 2:
                y_set.add(bv)
            else:
                x_set.add(bv)
    else:
        for bn, bv in box.items():
            if "x" in bn:
                x_set.add(bv)
            else:
                y_set.add(bv)

    left, top, right, bottom = (min(x_set), min(y_set), max(x_set), max(y_set))
    return [left, top, right, bottom]


def normalize_bbox(box: List[int], size: Tuple) -> List[int]:
    """Apply normalization to a bounding box. Values are normalized to [0, 1000].

    Parameters
    ----------
    box : List[int]
        Bounding box to be normalized
    size : Tuple
        Image size, (width, height)

    Returns
    -------
    List[int]
        The normalized bounding box.
    """

    def clip(min_num, num, max_num):
        return min(max(num, min_num), max_num)

    x0, y0, x1, y1 = box
    width, height = size

    x0 = clip(0, int((x0 / width) * 1000), 1000)
    y0 = clip(0, int((y0 / height) * 1000), 1000)
    x1 = clip(0, int((x1 / width) * 1000), 1000)
    y1 = clip(0, int((y1 / height) * 1000), 1000)
    assert x1 >= x0
    assert y1 >= y0
    return [x0, y0, x1, y1]


def merge_bbox(bbox_list: List[List[int]]) -> List[int]:
    """Merge a list of bounding boxes into one bounding box.

    Parameters
    ----------
    bbox_list : List[List[int]]
        List of bounding boxes to be merged.

    Returns
    -------
    List[int]
        The merged bounding box.
    """
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]


def sort_boxes(sample: List[List[int]]) -> List[int]:
    """Sort a list of bounding boxes by left-top to right-bottom order.
    Boxes with a vertical distance less than the average height of all boxes
    will be considered as the same line and sorted by left to right order.
    Return the index of sorted bounding boxes.

    Parameters
    ----------
    sample : List[List[int]]
        List of bounding boxes to be sorted.

    Returns
    -------
    List[int]
        The index of sorted bounding boxes.
    """

    if len(sample) == 0:
        return []

    sample = np.array(sample)
    p_x = (sample[:, 0] + sample[:, 2]) / 2.0
    p_y = (sample[:, 1] + sample[:, 3]) / 2.0
    m_h = np.sum(sample[:, 3] - sample[:, 1])
    m_h /= 2.0 * float(len(sample))
    sort_y = np.argsort(p_y)
    line = 0
    sort_y_c = [0]
    for i in range(1, sort_y.shape[0]):
        if (p_y[sort_y[i]] - p_y[sort_y[i - 1]]) < m_h:
            sort_y_c.append(line)
        else:
            line += 1
            sort_y_c.append(line)
    sort_y_c = np.asarray(sort_y_c)
    for i in range(0, max(sort_y_c) + 1):
        start = np.where(sort_y_c == i)[0][0]
        end = start + sum(sort_y_c == i)
        sort_y[start:end] = (sort_y[start:end])[np.argsort(p_x[sort_y[start:end]])]

    return sort_y.tolist()


def box_augmentation(bbox: Union[List, Tuple], image_w: int, image_h: int) -> Tuple:
    """Apply random jitter to a bounding box.

    Parameters
    ----------
    bbox : Tuple
        The bounding box to be jittered.
    image_w : int
        Image width
    image_h : int
        Image height

    Returns
    -------

    """
    left, top, right, bot = bbox

    x_dir = random.randint(0, 1)
    y_dir = random.randint(0, 1)

    x_move_ratio = random.randint(0, 10)
    y_move_ratio = random.randint(0, 30)
    x_move_dis = (right - left) * (x_move_ratio / 100)
    y_move_dis = (bot - top) * (y_move_ratio / 100)

    if x_dir:
        new_left = left + x_move_dis
        new_right = right + x_move_dis
    else:
        new_left = left - x_move_dis
        new_right = right - x_move_dis

    if y_dir:
        new_top = top + y_move_dis
        new_bot = bot + y_move_dis
    else:
        new_top = top + y_move_dis
        new_bot = bot + y_move_dis

    new_left, new_right = np.clip([new_left, new_right], 0, image_w)
    new_top, new_bot = np.clip([new_top, new_bot], 0, image_h)

    new_left = int(round(new_left))
    new_top = int(round(new_top))
    new_right = int(round(new_right))
    new_bot = int(round(new_bot))

    return new_left, new_top, new_right, new_bot


def string_f2h(text: str) -> str:
    """Convert full-width characters to half-width characters.

    Parameters
    ----------
    text : str
        The text to be converted.

    Returns
    -------
    str
        The converted text.
    """

    def char_f2h(char):
        code = ord(char)
        if code == 0x3000:
            return " "
        if 0xFF01 <= code <= 0xFF5E:
            return chr(code - 0xFEE0)
        return char

    return "".join(char_f2h(c) for c in text)
