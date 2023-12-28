def priority_queue_assignment(threshold, groups, queue):
    """
    threshold: priorities above threshold merge groups
    groups: Dict of { index:group }
    queue: Sorted list of items [(priority_ij, i, j)]
    """
    while len(queue) > 0:
        sim, i, j = queue.pop()
        if sim > threshold:
            groups[j] = groups[i]
        else:
            break

    return groups
