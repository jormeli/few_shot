import gin


@gin.configurable
def log(*args, verbose=True):
    if verbose:
        print(*args)
