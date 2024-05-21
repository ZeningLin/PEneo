from typing import Dict, List, Tuple, Union

import torch.distributed


def _calculate_linking_metric_core(pred: Union[Dict, List], gt: Union[Dict, List]):
    """Calculate the Precision, Recall, and F1 score for linking prediction

    Parameters
    ----------
    pred : Union[Dict, List]
        Prediction results, can be a dict mapped from head to tail,
        or a list of (head, tail) tuples
    gt : Union[Dict, List]
        Ground truth, can be a dict mapped from head to tail,
        or a list of (head, tail) tuples

    Returns
    -------
    Precision, Recall, F1 score,
    number of predictions, number of ground truth,
    and number of correct predictions
    """
    num_pred, num_gt, num_correct = 0.0, 0.0, 0.0
    if isinstance(pred, Dict):
        pred = [(k, v) for k, v in pred.items()]
    if isinstance(gt, Dict):
        gt = [(k, v) for k, v in gt.items()]
    num_pred += len(pred)
    num_gt += len(gt)
    for pred_item in pred:
        if pred_item in gt:
            num_correct += 1
    precision = num_correct / num_pred if num_pred > 0 else 0.0
    recall = num_correct / num_gt if num_gt > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1, num_pred, num_gt, num_correct


def _calculate_KV_metric_core(pred: List, gt: List, return_detail: bool = False):
    """Calculate the Precision, Recall, and F1 score for key-value pair prediction.
    Also generate detailed information for each prediction.

    Parameters
    ----------
    pred : List
        Prediction results
    gt : List
        Ground-truths
    return_detail : bool, optional
        If True, return detailed information for each prediction,
        by default False

    Returns
    -------
    Precision, Recall, F1 score,
    number of predictions, number of ground truth,
    number of correct predictions.
    If return_detail is True,
    also return detailed information for each prediction.
    """
    num_pred, num_gt, num_correct = 0.0, 0.0, 0.0
    detail = []
    matched_gt = []
    num_pred += len(pred)
    num_gt += len(gt)
    for p in pred:
        if p in gt:
            num_correct += 1
            if return_detail:
                detail.append({"status": "TP", "pred": p})
            matched_gt.append(p)
        else:
            if return_detail:
                detail.append({"status": "FP", "pred": p})
    precision = num_correct / len(pred) if len(pred) > 0 else 0.0
    recall = num_correct / len(gt) if len(gt) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    if return_detail:
        for g in gt:
            if g not in matched_gt:
                detail.append({"status": "FN", "gt": g})
        return precision, recall, f1, num_pred, num_gt, num_correct, detail

    return precision, recall, f1, num_pred, num_gt, num_correct


def calculate_KVPE_metric(
    all_pred: List[Tuple], all_gt: List[Tuple], all_fname: List[str]
):
    """Calculate the Precision, Recall, and F1 score for key-value pair prediction.

    Parameters
    ----------
    all_pred : List[Tuple]
        Prediction results parsed by the decoding module
    all_gt : List[Tuple]
        Ground-truths parsed by the decoding module
    all_fname : List[str]
        List of file names for detail information logging

    """
    sample_detail = []
    dist_file_result_list = []
    for fname, pred, gt in zip(all_fname, all_pred, all_gt):
        pred_kv_pairs, gt_kv_pairs = pred[0], gt[0]

        (
            sample_precision,
            sample_recall,
            sample_f1,
            sample_num_pred,
            sample_num_gt,
            sample_num_correct,
            sample_detail_info,
        ) = _calculate_KV_metric_core(pred_kv_pairs, gt_kv_pairs, return_detail=True)
        sample_detail.append(
            {
                "fname": fname,
                "num_pred": sample_num_pred,
                "num_gt": sample_num_gt,
                "num_correct": sample_num_correct,
                "precision": sample_precision,
                "recall": sample_recall,
                "f1": sample_f1,
                "detail": sample_detail_info,
            }
        )

        dist_file_result_list.append(
            [
                fname,
                sample_num_pred,
                sample_num_gt,
                sample_num_correct,
            ]
        )

    # gather all file results
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        world_size = torch.distributed.get_world_size()
        gather_file_result_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(
            gather_file_result_list, dist_file_result_list
        )
    else:
        gather_file_result_list = [dist_file_result_list]

    processed_fname = set()
    num_pred, num_gt, num_correct = 0.0, 0.0, 0.0
    num_sample_processed = 0
    gather_file_result_list = [
        item for sublist in gather_file_result_list for item in sublist
    ]
    for file_result in gather_file_result_list:
        (
            fname,
            sample_num_pred,
            sample_num_gt,
            sample_num_correct,
        ) = file_result
        if fname in processed_fname:
            # avoid duplication with distributed sampler
            continue

        processed_fname.add(fname)

        num_pred += sample_num_pred
        num_gt += sample_num_gt
        num_correct += sample_num_correct
        num_sample_processed += 1

    precision = num_correct / num_pred if num_pred > 0 else 0.0
    recall = num_correct / num_gt if num_gt > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    detail = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_pred": num_pred,
        "num_gt": num_gt,
        "num_correct": num_correct,
        "num_sample_processed": num_sample_processed,
        "detail": sample_detail,
    }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }, detail


def calculate_detail_KVPE_metric(
    all_pred: List[Tuple], all_gt: List[Tuple], all_fname: List[str]
):
    """Calculate the Precision, Recall, and F1 score for key-value pair prediction,
    line extraction, entity linking head/tail, and line grouping head/tail.

    Parameters
    ----------
    all_pred : List[Tuple]
        Prediction results parsed by the decoding module
    all_gt : List[Tuple]
        Ground-truths parsed by the decoding module
    all_fname : List[str]
        List of file names for detail information logging

    """
    sample_details, dist_file_result_list = [], []
    for fname, pred, gt in zip(all_fname, all_pred, all_gt):
        (
            pred_kv_pairs,
            pred_lines,
            _,
            pred_ent_linking_heads,
            pred_ent_linking_tails,
            pred_line_grouping_heads,
            pred_line_grouping_tails,
        ) = pred
        (
            gt_kv_pairs,
            gt_lines,
            _,
            gt_ent_linking_heads,
            gt_ent_linking_tails,
            gt_line_grouping_heads,
            gt_line_grouping_tails,
        ) = gt

        # kv-pair metric
        (
            sample_kv_precision,
            sample_kv_recall,
            sample_kv_f1,
            sample_kv_num_pred,
            sample_kv_num_gt,
            sample_kv_num_correct,
            sample_kv_detail_info,
        ) = _calculate_KV_metric_core(pred_kv_pairs, gt_kv_pairs, return_detail=True)

        # line extraction metric
        (
            sample_line_precision,
            sample_line_recall,
            sample_line_f1,
            sample_line_num_pred,
            sample_line_num_gt,
            sample_line_num_correct,
        ) = _calculate_KV_metric_core(pred_lines, gt_lines, return_detail=False)

        # entity linking head metric
        pred_ent_linking_heads = [(k, v) for k, v in pred_ent_linking_heads.items()]
        gt_ent_linking_heads = [(k, v) for k, v in gt_ent_linking_heads.items()]
        (
            sample_ent_linking_head_precision,
            sample_ent_linking_head_recall,
            sample_ent_linking_head_f1,
            sample_ent_linking_head_num_pred,
            sample_ent_linking_head_num_gt,
            sample_ent_linking_head_num_correct,
        ) = _calculate_linking_metric_core(pred_ent_linking_heads, gt_ent_linking_heads)

        # entity linking tail metric
        pred_ent_linking_tails = [(k, v) for k, v in pred_ent_linking_tails.items()]
        gt_ent_linking_tails = [(k, v) for k, v in gt_ent_linking_tails.items()]
        (
            sample_ent_linking_tail_precision,
            sample_ent_linking_tail_recall,
            sample_ent_linking_tail_f1,
            sample_ent_linking_tail_num_pred,
            sample_ent_linking_tail_num_gt,
            sample_ent_linking_tail_num_correct,
        ) = _calculate_linking_metric_core(pred_ent_linking_tails, gt_ent_linking_tails)

        # line grouping head metric
        pred_line_grouping_heads = [(k, v) for k, v in pred_line_grouping_heads.items()]
        gt_line_grouping_heads = [(k, v) for k, v in gt_line_grouping_heads.items()]
        (
            sample_line_grouping_head_precision,
            sample_line_grouping_head_recall,
            sample_line_grouping_head_f1,
            sample_line_grouping_head_num_pred,
            sample_line_grouping_head_num_gt,
            sample_line_grouping_head_num_correct,
        ) = _calculate_linking_metric_core(
            pred_line_grouping_heads, gt_line_grouping_heads
        )

        # line grouping tail metric
        pred_line_grouping_tails = [(k, v) for k, v in pred_line_grouping_tails.items()]
        gt_line_grouping_tails = [(k, v) for k, v in gt_line_grouping_tails.items()]
        (
            sample_line_grouping_tail_precision,
            sample_line_grouping_tail_recall,
            sample_line_grouping_tail_f1,
            sample_line_grouping_tail_num_pred,
            sample_line_grouping_tail_num_gt,
            sample_line_grouping_tail_num_correct,
        ) = _calculate_linking_metric_core(
            pred_line_grouping_tails, gt_line_grouping_tails
        )

        sample_details.append(
            {
                "fname": fname,
                "kv_pair": {
                    "num_pred": sample_kv_num_pred,
                    "num_gt": sample_kv_num_gt,
                    "num_correct": sample_kv_num_correct,
                    "precision": sample_kv_precision,
                    "recall": sample_kv_recall,
                    "f1": sample_kv_f1,
                },
                "line_extraction": {
                    "num_pred": sample_line_num_pred,
                    "num_gt": sample_line_num_gt,
                    "num_correct": sample_line_num_correct,
                    "precision": sample_line_precision,
                    "recall": sample_line_recall,
                    "f1": sample_line_f1,
                },
                "ent_linking_head": {
                    "num_pred": sample_ent_linking_head_num_pred,
                    "num_gt": sample_ent_linking_head_num_gt,
                    "num_correct": sample_ent_linking_head_num_correct,
                    "precision": sample_ent_linking_head_precision,
                    "recall": sample_ent_linking_head_recall,
                    "f1": sample_ent_linking_head_f1,
                },
                "ent_linking_tail": {
                    "num_pred": sample_ent_linking_tail_num_pred,
                    "num_gt": sample_ent_linking_tail_num_gt,
                    "num_correct": sample_ent_linking_tail_num_correct,
                    "precision": sample_ent_linking_tail_precision,
                    "recall": sample_ent_linking_tail_recall,
                    "f1": sample_ent_linking_tail_f1,
                },
                "line_grouping_head": {
                    "num_pred": sample_line_grouping_head_num_pred,
                    "num_gt": sample_line_grouping_head_num_gt,
                    "num_correct": sample_line_grouping_head_num_correct,
                    "precision": sample_line_grouping_head_precision,
                    "recall": sample_line_grouping_head_recall,
                    "f1": sample_line_grouping_head_f1,
                },
                "line_grouping_tail": {
                    "num_pred": sample_line_grouping_tail_num_pred,
                    "num_gt": sample_line_grouping_tail_num_gt,
                    "num_correct": sample_line_grouping_tail_num_correct,
                    "precision": sample_line_grouping_tail_precision,
                    "recall": sample_line_grouping_tail_recall,
                    "f1": sample_line_grouping_tail_f1,
                },
                "detail": sample_kv_detail_info,
            }
        )

        dist_file_result_list.append(
            [
                fname,
                sample_kv_num_pred,
                sample_kv_num_gt,
                sample_kv_num_correct,
                sample_line_num_pred,
                sample_line_num_gt,
                sample_line_num_correct,
                sample_ent_linking_head_num_pred,
                sample_ent_linking_head_num_gt,
                sample_ent_linking_head_num_correct,
                sample_ent_linking_tail_num_pred,
                sample_ent_linking_tail_num_gt,
                sample_ent_linking_tail_num_correct,
                sample_line_grouping_head_num_pred,
                sample_line_grouping_head_num_gt,
                sample_line_grouping_head_num_correct,
                sample_line_grouping_tail_num_pred,
                sample_line_grouping_tail_num_gt,
                sample_line_grouping_tail_num_correct,
            ]
        )

    # gather all file results
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        world_size = torch.distributed.get_world_size()
        gather_file_result_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(
            gather_file_result_list, dist_file_result_list
        )
    else:
        gather_file_result_list = [dist_file_result_list]

    processed_fname = set()
    num_kv_pred, num_kv_gt, num_kv_correct = 0.0, 0.0, 0.0
    num_line_pred, num_line_gt, num_line_correct = 0.0, 0.0, 0.0
    num_ent_linking_head_pred, num_ent_linking_head_gt, num_ent_linking_head_correct = (
        0.0,
        0.0,
        0.0,
    )
    num_ent_linking_tail_pred, num_ent_linking_tail_gt, num_ent_linking_tail_correct = (
        0.0,
        0.0,
        0.0,
    )
    (
        num_line_grouping_head_pred,
        num_line_grouping_head_gt,
        num_line_grouping_head_correct,
    ) = (
        0.0,
        0.0,
        0.0,
    )
    (
        num_line_grouping_tail_pred,
        num_line_grouping_tail_gt,
        num_line_grouping_tail_correct,
    ) = (
        0.0,
        0.0,
        0.0,
    )
    num_sample_processed = 0
    gather_file_result_list = [
        item for sublist in gather_file_result_list for item in sublist
    ]
    for file_result in gather_file_result_list:
        (
            fname,
            sample_kv_num_pred,
            sample_kv_num_gt,
            sample_kv_num_correct,
            sample_line_num_pred,
            sample_line_num_gt,
            sample_line_num_correct,
            sample_ent_linking_head_num_pred,
            sample_ent_linking_head_num_gt,
            sample_ent_linking_head_num_correct,
            sample_ent_linking_tail_num_pred,
            sample_ent_linking_tail_num_gt,
            sample_ent_linking_tail_num_correct,
            sample_line_grouping_head_num_pred,
            sample_line_grouping_head_num_gt,
            sample_line_grouping_head_num_correct,
            sample_line_grouping_tail_num_pred,
            sample_line_grouping_tail_num_gt,
            sample_line_grouping_tail_num_correct,
        ) = file_result
        if fname in processed_fname:
            # avoid duplication with distributed sampler
            continue

        processed_fname.add(fname)

        num_kv_pred += sample_kv_num_pred
        num_kv_gt += sample_kv_num_gt
        num_kv_correct += sample_kv_num_correct

        num_line_pred += sample_line_num_pred
        num_line_gt += sample_line_num_gt
        num_line_correct += sample_line_num_correct

        num_ent_linking_head_pred += sample_ent_linking_head_num_pred
        num_ent_linking_head_gt += sample_ent_linking_head_num_gt
        num_ent_linking_head_correct += sample_ent_linking_head_num_correct

        num_ent_linking_tail_pred += sample_ent_linking_tail_num_pred
        num_ent_linking_tail_gt += sample_ent_linking_tail_num_gt
        num_ent_linking_tail_correct += sample_ent_linking_tail_num_correct

        num_line_grouping_head_pred += sample_line_grouping_head_num_pred
        num_line_grouping_head_gt += sample_line_grouping_head_num_gt
        num_line_grouping_head_correct += sample_line_grouping_head_num_correct

        num_line_grouping_tail_pred += sample_line_grouping_tail_num_pred
        num_line_grouping_tail_gt += sample_line_grouping_tail_num_gt
        num_line_grouping_tail_correct += sample_line_grouping_tail_num_correct

        num_sample_processed += 1

    kv_precision = num_kv_correct / num_kv_pred if num_kv_pred > 0 else 0.0
    kv_recall = num_kv_correct / num_kv_gt if num_kv_gt > 0 else 0.0
    kv_f1 = (
        (2 * kv_precision * kv_recall) / (kv_precision + kv_recall)
        if kv_precision + kv_recall > 0
        else 0.0
    )
    line_precision = num_line_correct / num_line_pred if num_line_pred > 0 else 0.0
    line_recall = num_line_correct / num_line_gt if num_line_gt > 0 else 0.0
    line_f1 = (
        (2 * line_precision * line_recall) / (line_precision + line_recall)
        if line_precision + line_recall > 0
        else 0.0
    )
    ent_linking_head_precision = (
        num_ent_linking_head_correct / num_ent_linking_head_pred
        if num_ent_linking_head_pred > 0
        else 0.0
    )
    ent_linking_head_recall = (
        num_ent_linking_head_correct / num_ent_linking_head_gt
        if num_ent_linking_head_gt > 0
        else 0.0
    )
    ent_linking_head_f1 = (
        (2 * ent_linking_head_precision * ent_linking_head_recall)
        / (ent_linking_head_precision + ent_linking_head_recall)
        if ent_linking_head_precision + ent_linking_head_recall > 0
        else 0.0
    )
    ent_linking_tail_precision = (
        num_ent_linking_tail_correct / num_ent_linking_tail_pred
        if num_ent_linking_tail_pred > 0
        else 0.0
    )
    ent_linking_tail_recall = (
        num_ent_linking_tail_correct / num_ent_linking_tail_gt
        if num_ent_linking_tail_gt > 0
        else 0.0
    )
    ent_linking_tail_f1 = (
        (2 * ent_linking_tail_precision * ent_linking_tail_recall)
        / (ent_linking_tail_precision + ent_linking_tail_recall)
        if ent_linking_tail_precision + ent_linking_tail_recall > 0
        else 0.0
    )
    line_grouping_head_precision = (
        num_line_grouping_head_correct / num_line_grouping_head_pred
        if num_line_grouping_head_pred > 0
        else 0.0
    )
    line_grouping_head_recall = (
        num_line_grouping_head_correct / num_line_grouping_head_gt
        if num_line_grouping_head_gt > 0
        else 0.0
    )
    line_grouping_head_f1 = (
        (2 * line_grouping_head_precision * line_grouping_head_recall)
        / (line_grouping_head_precision + line_grouping_head_recall)
        if line_grouping_head_precision + line_grouping_head_recall > 0
        else 0.0
    )
    line_grouping_tail_precision = (
        num_line_grouping_tail_correct / num_line_grouping_tail_pred
        if num_line_grouping_tail_pred > 0
        else 0.0
    )
    line_grouping_tail_recall = (
        num_line_grouping_tail_correct / num_line_grouping_tail_gt
        if num_line_grouping_tail_gt > 0
        else 0.0
    )
    line_grouping_tail_f1 = (
        (2 * line_grouping_tail_precision * line_grouping_tail_recall)
        / (line_grouping_tail_precision + line_grouping_tail_recall)
        if line_grouping_tail_precision + line_grouping_tail_recall > 0
        else 0.0
    )

    detail = {
        "kv_pair": {
            "precision": kv_precision,
            "recall": kv_recall,
            "f1": kv_f1,
            "num_pred": num_kv_pred,
            "num_gt": num_kv_gt,
            "num_correct": num_kv_correct,
        },
        "line_extraction": {
            "precision": line_precision,
            "recall": line_recall,
            "f1": line_f1,
            "num_pred": num_line_pred,
            "num_gt": num_line_gt,
            "num_correct": num_line_correct,
        },
        "ent_linking_head": {
            "precision": ent_linking_head_precision,
            "recall": ent_linking_head_recall,
            "f1": ent_linking_head_f1,
            "num_pred": num_ent_linking_head_pred,
            "num_gt": num_ent_linking_head_gt,
            "num_correct": num_ent_linking_head_correct,
        },
        "ent_linking_tail": {
            "precision": ent_linking_tail_precision,
            "recall": ent_linking_tail_recall,
            "f1": ent_linking_tail_f1,
            "num_pred": num_ent_linking_tail_pred,
            "num_gt": num_ent_linking_tail_gt,
            "num_correct": num_ent_linking_tail_correct,
        },
        "line_grouping_head": {
            "precision": line_grouping_head_precision,
            "recall": line_grouping_head_recall,
            "f1": line_grouping_head_f1,
            "num_pred": num_line_grouping_head_pred,
            "num_gt": num_line_grouping_head_gt,
            "num_correct": num_line_grouping_head_correct,
        },
        "line_grouping_tail": {
            "precision": line_grouping_tail_precision,
            "recall": line_grouping_tail_recall,
            "f1": line_grouping_tail_f1,
            "num_pred": num_line_grouping_tail_pred,
            "num_gt": num_line_grouping_tail_gt,
            "num_correct": num_line_grouping_tail_correct,
        },
        "detail": sample_details,
    }

    return {
        "precision": kv_precision,
        "recall": kv_recall,
        "f1": kv_f1,
        "line_extraction_precision": line_precision,
        "line_extraction_recall": line_recall,
        "line_extraction_f1": line_f1,
        "ent_linking_head_precision": ent_linking_head_precision,
        "ent_linking_head_recall": ent_linking_head_recall,
        "ent_linking_head_f1": ent_linking_head_f1,
        "ent_linking_tail_precision": ent_linking_tail_precision,
        "ent_linking_tail_recall": ent_linking_tail_recall,
        "ent_linking_tail_f1": ent_linking_tail_f1,
        "line_grouping_head_precision": line_grouping_head_precision,
        "line_grouping_head_recall": line_grouping_head_recall,
        "line_grouping_head_f1": line_grouping_head_f1,
        "line_grouping_tail_precision": line_grouping_tail_precision,
        "line_grouping_tail_recall": line_grouping_tail_recall,
        "line_grouping_tail_f1": line_grouping_tail_f1,
    }, detail
