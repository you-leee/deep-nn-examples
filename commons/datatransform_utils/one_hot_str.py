from . import np


def one_hot_str(vocab, text, as_matrix = True):
    """
    Arguments:
    vocab -- set of words used in texts
    text -- the text which will be encoded

    Returns:
    one_hot -- one hot matrix/vector
    """

    words = text.split()
    vocab_dict = {e:i for i, e in enumerate(vocab)} # encoding of words
    one_hot = [vocab_dict[w] for w in words if vocab_dict.get(w) != None]

    if len(one_hot) == 0:
        return []

    if as_matrix:
        one_hot_vec = one_hot
        one_hot = np.zeros((len(one_hot_vec), len(vocab)))
        one_hot[np.arange(len(one_hot_vec)), one_hot_vec] = 1

    return one_hot
