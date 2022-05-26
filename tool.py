def time_format(duration):

    hours    = int( duration                                / 3600)
    minutes  = int((duration - hours * 3600)                /   60)
    seconds  = int( duration - hours * 3600 - minutes * 60        )
    time_str = f'{hours:>2d}h{minutes:>2d}m{seconds:>2d}s'

    return time_str
