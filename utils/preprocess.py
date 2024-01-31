def img_aligned_scale(target_size):
    for i in range(20):
        if 2 ** i >= target_size:
            return 2 ** i
    return target_size
