import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RecursiveXYCut:
    """Implementation of XY-cut algorithm for splitting and sorting bounding boxes."""

    @staticmethod
    def _get_split_pos(
        projection: np.ndarray, min_value: float, min_gap: float
    ) -> Tuple[np.ndarray]:
        index = np.where(projection > min_value)[0]
        if not len(index):
            return

        diff = index[1:] - index[0:-1]
        diff_index = np.where(diff > min_gap)[0]
        zero_intvl_start = index[diff_index]
        zero_intvl_end = index[diff_index + 1]

        # convert to index of projection range:
        # the start index of zero interval is the end index of projection
        start = np.insert(zero_intvl_end, 0, index[0])
        end = np.append(zero_intvl_start, index[-1])
        end += 1  # end index will be excluded as index slice

        return start, end

    @staticmethod
    def projection_x(bboxes: List[List[int]], width: int) -> np.ndarray:
        projection = np.zeros(width)
        for bbox in bboxes:
            x0, _, x1, _ = bbox
            projection[x0:x1] += 1
        return projection

    @staticmethod
    def projection_y(bboxes: List[List[int]], height: int) -> np.ndarray:
        projection = np.zeros(height)
        for bbox in bboxes:
            _, y0, _, y1 = bbox
            projection[y0:y1] += 1
        return projection

    @staticmethod
    def check_bbox_validation(bboxes: List[List[int]], width: int, height: int) -> bool:
        for bbox in bboxes:
            x0, y0, x1, y1 = bbox
            if not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height):
                logger.error(
                    f"invalid_box: [{x0}, {y0}, {x1}, {y1}], image_size: ({width}, {height})"
                )
                return False
        return True

    @staticmethod
    def recursive_xy_cut(
        bboxes: List[List[int]],
        width: int,
        height: int,
        min_w: float = 0.0,
        min_h: float = 0.0,
        min_dx: float = 5.0,
        min_dy: float = 5.0,
    ):
        """Split image with recursive xy-cut algorithm.

        Parameters
        ----------
        bboxes : List[List[int]]
            List of bounding boxes to be sorted.
        width : int
            Image width
        height : int
            Image height
        min_w : float
            Ignore bbox if the width is less than this value.
        min_h : float
            Ignore bbox if the height is less than this value.
        min_dx : float
            Merge two bboxes if the x-gap is less than this value.
        min_dy : float
            Merge two bboxes if the y-gap is less than this value.

        Returns
        -------
        List[int]
            The index of sorted bounding boxes.
        """
        if not RecursiveXYCut.check_bbox_validation(bboxes, width, height):
            raise ValueError("Invalid bbox!")

        def xy_cut(bboxes: List[List[int]], index: List[int]):
            # cut along y-direction
            projection_y = RecursiveXYCut.projection_y(bboxes, height)
            pos_y = RecursiveXYCut._get_split_pos(projection_y, min_w, min_dy)

            # cut along x-direction for each part
            arr_y0, arr_y1 = pos_y  # start and end
            for r0, r1 in zip(arr_y0, arr_y1):
                bboxes_y_cut, index_y_cut = zip(
                    *filter(
                        lambda x: r0 <= 0.5 * (x[0][1] + x[0][3]) < r1,
                        zip(bboxes, index),
                    )
                )

                if len(bboxes_y_cut) == 1:
                    new_index.extend(index_y_cut)
                    continue

                projection_x = RecursiveXYCut.projection_x(bboxes_y_cut, width)
                pos_x = RecursiveXYCut._get_split_pos(projection_x, min_h, min_dx)

                arr_x0, arr_x1 = pos_x
                if len(arr_x0) == 1:
                    if len(index_y_cut) >= 2:
                        candidate_boxes = [org_bboxes[i] for i in index_y_cut]
                        project_x = RecursiveXYCut.projection_x(candidate_boxes, width)
                        project_y = RecursiveXYCut.projection_y(candidate_boxes, height)
                        overlap_x = (
                            project_x[project_x >= 2].sum() / (project_x >= 1).sum()
                        )
                        overlap_y = (
                            project_y[project_y >= 2].sum() / (project_y >= 1).sum()
                        )
                        candidate_boxes = [(i, org_bboxes[i]) for i in index_y_cut]
                        if overlap_y > overlap_x:
                            candidate_boxes.sort(key=lambda x: sum(x[1][0::2]) / 2)
                        else:
                            candidate_boxes.sort(key=lambda x: sum(x[1][1::2]) / 2)
                        index_y_cut = (i[0] for i in candidate_boxes)
                    old_len = len(new_index)
                    new_index.extend(index_y_cut)
                    continue

                # xy-cut recursively if the count of blocks > 1
                for c0, c1 in zip(arr_x0, arr_x1):
                    bboxes_yx_cut, index_yx_cut = zip(
                        *filter(
                            lambda x: c0 <= 0.5 * (x[0][0] + x[0][2]) < c1,
                            zip(bboxes_y_cut, index_y_cut),
                        )
                    )

                    if len(bboxes_yx_cut) == 1:
                        old_len = len(new_index)
                        new_index.extend(index_yx_cut)
                        continue

                    xy_cut(bboxes_yx_cut, index_yx_cut)

        # do xy-cut recursively
        new_index = []
        org_bboxes = bboxes
        xy_cut(bboxes=bboxes, index=list(range(len(bboxes))))
        return new_index
