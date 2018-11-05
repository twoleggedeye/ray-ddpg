def ewma(current, new, alpha=0.8):
    if current is None:
        return new
    else:
        return (1 - alpha) * current + alpha * new


def interpolate_params(source, target, tau=0.001):
    new_state_dict = {}
    target_state_dict = target.state_dict()
    source_state_dict = source.state_dict()
    for k in source_state_dict.keys():
        new_state_dict[k] = tau * source_state_dict[k] + (1 - tau) * target_state_dict[k]
    target.load_state_dict(new_state_dict) 


def get_cpu_state_dict(model):
    return {k: v.cpu() for k, v in model.state_dict().items()}
