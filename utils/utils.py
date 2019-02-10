import torch


def get_class_weights(dset_name='sun'):
    with open('./text/weights.txt', 'r') as f:
        lines = f.read().splitlines()
    dset_weights, read = [], False

    for line in lines:
        if line.lower().find(dset_name) is not -1 and not read:
            read = True
            continue
        if line == '-':
            read = False
        if read:
            weights = line.split(', ')
            for weight in weights:
                dset_weights.append(float(weight))
    return torch.cuda.FloatTensor(dset_weights)


def print_time_info(start_time, end_time):
    print('[INFO] Start and end time of the last session: %s - %s'
          % (start_time.strftime('%d.%m.%Y %H:%M:%S'), end_time.strftime('%d.%m.%Y %H:%M:%S')))
    print('[INFO] Total time previous session took:', (end_time - start_time), '\n')
