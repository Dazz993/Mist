def get_add_shape(x_shape, y_shape):
    """Get the shape of the result of adding two tensors."""
    if len(x_shape) == 0:
        return y_shape
    elif len(y_shape) == 0:
        return x_shape
    else:
        max_length = max(len(x_shape), len(y_shape))
        x_shape = [1] * (max_length - len(x_shape)) + list(x_shape)
        y_shape = [1] * (max_length - len(y_shape)) + list(y_shape)
        shape = []
        for x, y in zip(x_shape, y_shape):
            if x == 1:
                shape.append(y)
            elif y == 1:
                shape.append(x)
            elif x == y:
                shape.append(x)
            else:
                raise ValueError(
                    f"operands could not be broadcast together with shapes {x_shape} {y_shape}"
                )

        return tuple(shape)


def get_matmul_shape(x_shape, y_shape):
    if len(x_shape) < 2 or len(y_shape) < 2:
        raise ValueError("Input tensors must have at least 2 dimensions")
    if x_shape[-1] != y_shape[-2]:
        raise ValueError(
            f"Input tensors have incompatible shapes: {x_shape} and {y_shape}"
        )
    pre_shape = get_add_shape(x_shape[:-2], y_shape[:-2])
    mm_shape = (x_shape[-2], y_shape[-1])
    return pre_shape + mm_shape
