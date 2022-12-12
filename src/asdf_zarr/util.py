def iter_keys(shape, chunk_shape):
    if len(shape) == 1:
        for i in range(0, shape[0], chunk_shape[0]):
            yield str(int(i / chunk_shape[0]))
    else:
        for i in range(0, shape[0], chunk_shape[0]):
            for key in iter_keys(shape[1:], chunk_shape[1:]):
                yield str(int(i / chunk_shape[0])) + "." + key


def get_at_index(sequence, key):
    for idx in [int(k) for k in key.split(".")]:
        sequence = sequence[idx]
    return sequence


def set_at_index(sequence, key, value):
    indices = [int(k) for k in key.split(".")]
    for idx in indices[:-1]:
        sequence = sequence[idx]
    sequence[indices[-1]] = value
