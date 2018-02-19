import numpy as np

def _smooth_posteriors(posteriors, smoothed, j, w_smooth):
    '''
    p'[i,j] = 1 / (j - h_smooth + 1) * sum(p[i,k], k=h_smooth, end=j)
    h_smooth = max(1, j - w_smooth + 1)
    '''
    h_smooth = max(0, j-w_smooth+1)

    coef = 1 / (j - h_smooth + 1)
    smoothed[:, j] += coef * np.sum(posteriors[:, h_smooth:j+1], axis=1)
    return smoothed


def _calc_confidence(smoothed_posteriors, j, w_max, num_labels):
    '''
    confidence score at jth frame
    c = (n-1)_root(mult(max(p'[i,k], start=h_max, end=j), i=1, end=n-1))
    h_max = max(1, j - w_max + 1)
    '''
    h_max = max(1, j-w_max+1)
    max_val = np.amax(smoothed_posteriors[1:, h_max:j+1], axis=1)

    confidence = np.prod(max_val)
    confidence = np.power(confidence, 1/(num_labels-1))
    return confidence


def recommend_word(net_out, smoothed_out, frame, w_smooth, w_max, threshold):
    num_labels = net_out.shape[0]

    smoothed_out = _smooth_posteriors(net_out, smoothed_out, frame, w_smooth)

    if frame < 10:
        return smoothed_out, -1, 0

    conf = _calc_confidence(smoothed_out, frame, w_max, num_labels)
    if conf < threshold:
        return smoothed_out, -1, conf

    word = np.argmax(smoothed_out[:, frame])
    return smoothed_out, word, conf


"""
'''   TESTING   '''
x = np.array([[.45, .14, .09, .05],
              [.4,  .60, .76, .89],
              [.15, .26, .15, .06]], dtype=float)
s = np.zeros_like(x, dtype=float)

for i in range(len(x[0])):
    s, word, conf = recommend_word(x, s, i, 10, 10, 0)
    print(s)
    print(word, conf)
"""

