import cvxpy as cp
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from measure_latency import latency_focas_fit
from scipy import stats
import argparse

# quality scores of features from different blocks, normalized to 0-1
# original PSNR scores: 31.03, 31.47, 31.83, 32.18, 32.62, 32.97, 33.32, 33.83, 34.70, 36.39
q = [0, 0.0819, 0.1489, 0.2141, 0.29608, 0.3612, 0.4264, 0.5214, 0.6834, 1]


# measure latencies of 108 sampled FOCAS models
measured_latency = latency_focas_fit()
print("measured latency")
print(measured_latency)

# fit a latency estimation model using linear regression
sum_region_list = []
latency_list = []
for element in measured_latency:
    b1, b2, b3, s1, s2, latency = element
    sum_region = b1*270*480+(b2-b1)*s1*s1/16+(b3-b2)*s2*s2/16
    sum_region_list.append(sum_region)
    latency_list.append(latency * 10**3) # unit: ms

A, B, _, _, _ = stats.linregress(sum_region_list, latency_list) # intercept, slope
print("A(slope)",A,"B(intercept)", B)


# compute fovaeted visual quality score
def compute_quality(blocks, d):
    H = 1080
    W = 1920
    h_eye = H // 2
    w_eye = W // 2
    sigma_h = 64
    sigma_w = 64

    regions = []
    for ele in d:
        size = int(ele)//2*2*4
        regions.append((size,size))
    regions.append((H,W))
    regions.reverse()
    print("blocks", blocks, "regions", regions)

    h_mat = np.arange(H).reshape((H, 1)).repeat(W, axis=1)
    w_mat = np.arange(W).reshape((1, W)).repeat(H, axis=0)
    weight = np.exp(-((h_mat - h_eye) ** 2) / (2 * sigma_h ** 2) - ((w_mat - w_eye) ** 2) / (2 * sigma_w ** 2))
    weight /= 2 * np.pi * sigma_h * sigma_w

    quality = np.zeros((H, W))
    for i in range(len(regions)):
        q_sub = q[blocks[i]-1]
        HH = regions[i][0]//2
        WW = regions[i][1]//2
        quality[h_eye-HH:h_eye+HH, w_eye-WW:w_eye+WW] = q_sub * np.ones((HH*2,WW*2))

    score = np.sum(weight*quality)
    return score, blocks, regions


# quality allocation solver
def solve_region(blocks, T, L_s=270, L_W_s = 480):
    blocks = deepcopy(blocks)
    blocks.reverse()
    len_a = len(blocks) - 1
    a = cp.Variable(len_a)

    q_diff = []
    for i in range(len_a):
        q_diff.append(q[blocks[i]-1] - q[blocks[i+1]-1])
    q_diff = np.array(q_diff)

    obj = cp.Minimize(cp.sum(cp.multiply(q_diff, cp.exp(-a / (2*64*64) ))))

    constraints = []
    blocks.append(0)

    b_diff = []
    for i in range(len_a):
       b_diff.append(blocks[i] - blocks[i+1])
    b_diff = np.array(b_diff)
    constraints.append(cp.sum(cp.multiply(b_diff,a)) + (blocks[len_a]-blocks[len_a+1])*L_s*L_W_s <= (T-B) / A)
    constraints.append(a[0] >= 0)
    constraints.append(a[-1] <= L_s*L_s)
    for i in range(len_a-1):
        constraints.append(a[i] <= a[i+1])

    prob = cp.Problem(obj, constraints)
    prob.solve()
    value = prob.value
    print("blocks", blocks)
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal variable", a.value)
    if prob.status == "optimal" or prob.status == "optimal_inaccurate":
        d = np.sqrt(np.array(a.value))
        latency = B + A * np.sum(b_diff * d * d) + A * (blocks[len_a]-blocks[len_a+1])*L_s*L_W_s
        print("estimated latency", latency)
        return value, d, latency
    else:
        return None, None, None


def solve_3region(time_limit):
    max_block = 10
    opt_quality = 0
    opt_block = None
    opt_region = None
    opt_latency = None

    for i in range(max_block-2):
        for j in range(i+1, max_block-1):
            for k in range(j+1, max_block):
                blocks = [i+1, j+1, k+1]
                value, d, latency = solve_region(blocks, time_limit)
                if value is not None:
                    quality, block, region = compute_quality(blocks, d)
                    print("quality", quality)
                    if quality > opt_quality:
                        opt_quality = quality
                        opt_block = block
                        opt_region = region
                        opt_latency = latency

    print("--------- optimal solution ---------")
    print("blocks", opt_block)
    print("regions", opt_region)
    print("estimated latency", opt_latency)
    print("estimated quality", opt_quality)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latency', default=25, type=float)
    opt = parser.parse_args()

    solve_3region(time_limit=opt.time)


