import torch
import string


def onehotEncodeLabel(label):
    '''
    Onehot encode label string into 5 * 36 
    Args:
        label: alphanumeric (all upper case) string of length 5, e.g. '1A2B3'
    '''
    onehot = torch.zeros(5)
    for i, c in enumerate(label):
        onehot[i] = int(c, 36)
    return onehot

def batchOnehotEncodeLabel(labels):
    '''
    Onehot encode a batch of labels
    Args:
        labels: a list of alphanumeric strings of length 5
    '''
    return torch.stack([onehotEncodeLabel(label) for label in labels])

def int_to_base36(n):
    if 0 <= n <= 9:
        return str(n)  # Digits 0-9 stay the same
    elif 10 <= n <= 35:
        return string.ascii_uppercase[n - 10]  # Convert 10-35 to 'A'-'Z'
    else:
        raise ValueError("Number out of range for a single base-36 digit")

def onehotDecodeLabel(onehot):
    '''
    Decode onehot encoded label into alphanumeric string
    Args:
        onehot: tensor of shape (5, 36)
    '''
    return ''.join([int_to_base36(int(torch.argmax(onehot[i]))) for i in range(5)])

def batchOnehotDecodeLabel(onehots):
    '''
    Decode a batch of onehot encoded labels
    Args:
        onehots: tensor of shape (batch_size, 5, 36)
    '''
    return [onehotDecodeLabel(onehot) for onehot in onehots]

def decodeCTCOutput(output):
    '''
    Decode CTC output into alphanumeric strings
    Args:
        output: tensor of shape (18, 37)
    '''
    # get the most probable character for each time step
    output = torch.argmax(output, dim=1) # (18)
    # remove duplicates
    decoded = []
    prev_c = None
    for c in output:
        if c != 0 and c != prev_c: # not space and not character
            decoded.append(int_to_base36(c.item() - 1))
        prev_c = c
    return ''.join(decoded)

def batchDecodeCTCOutput(outputs):
    '''
    Decode a batch of CTC outputs
    Args:
        outputs: tensor of shape (batch_size, 18, 37)
    '''
    return [decodeCTCOutput(output) for output in outputs]