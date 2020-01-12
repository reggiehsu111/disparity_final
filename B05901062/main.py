import numpy as np
import cv2
import time
import multiprocessing as mp
from util import readPFM, writePFM
import argparse

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./TLD0.pfm', type=str, help='left disparity map')

N = 8192
WINDOW_SIZE = 37
SIGMA = 6
h, w, ch, max_disp, pad = 0, 0, 0, 0, 20
Il_gray, Ir_gray, Il_lab, lr_lab, dx1, dy1, dx2, dy2 = [], [], [], [], [], [], [], []
costl, costr, phi_l, phi_r, labels_l, labels_r = [], [], [], [], [], []

def compute_cost(i):
    costl = np.zeros((w, N), dtype=np.bool)
    costr = np.zeros((w, N), dtype=np.bool)
    phi_l = np.zeros((w, N), dtype=np.bool)
    phi_r = np.zeros((w, N), dtype=np.bool)
    px = np.clip(i + dx1, max(i - WINDOW_SIZE // 2, 0), min(i + WINDOW_SIZE // 2, h - 1))
    qx = np.clip(i + dx2, max(i - WINDOW_SIZE // 2, 0), min(i + WINDOW_SIZE // 2, h - 1))
    for j in range(w):
        py = np.clip(j + dy1, max(j - WINDOW_SIZE // 2, 0), min(j + WINDOW_SIZE // 2, w - 1))
        qy = np.clip(j + dy2, max(j - WINDOW_SIZE // 2, 0), min(j + WINDOW_SIZE // 2, w - 1))
        costl[j] = (Il_gray[px, py] > Il_gray[qx, qy])
        costr[j] = (Ir_gray[px, py] > Ir_gray[qx, qy])
        Wl = np.amax(
            np.array([
                np.sum(np.abs(Il_lab[i, j] - Il_lab[px, py]), axis=1),
                np.sum(np.abs(Il_lab[i, j] - Il_lab[qx, qy]), axis=1)
            ]),
            axis=0
        )
        T = np.percentile(Wl, 25)
        phi_l[j] = (Wl <= T)
        Wr = np.amax(
            np.array([
                np.sum(np.abs(Ir_lab[i, j] - Ir_lab[px, py]), axis=1),
                np.sum(np.abs(Ir_lab[i, j] - Ir_lab[qx, qy]), axis=1)
            ]),
            axis=0
        )
        T = np.percentile(Wr, 25)
        phi_r[j] = (Wr <= T)
    return costl, costr, phi_l, phi_r

def find_disp(i):
    labels_l = np.zeros(w)
    labels_r = np.zeros(w)
    dis_l = np.zeros((w, max_disp))
    for j in range(w):
        min_dis = float('Inf')
        for d in range(max_disp):
            if j - d < 0:
                dis_l[j, d] = N
                continue
            tmp = (costl[i, j] != costr[i, j - d]) & (phi_l[i, j] == True)
            dis = np.sum(tmp)
            dis_l[j, d] = dis
            if dis < min_dis:
                min_dis = dis
                labels_l[j] = d
        min_dis = float('Inf')        
        for d in range(max_disp):
            if j + d >= w:
                break
            tmp = (costl[i, j + d] != costr[i, j]) & (phi_r[i, j] == True)
            dis = np.sum(tmp)
            if dis < min_dis:
                min_dis = dis
                labels_r[j] = d
    return labels_l, labels_r, dis_l

def check_validity(disp_l, disp_r):
#         h, w = left_disparity_map.shape
    mismatch_range = int(max_disp / 8)
    validity_map = np.zeros((h, w), dtype = 'int8')  
    disp = np.zeros((h, w))
    for x in range(w):
        for y in range(h):
            disp[y, x] = disp_l[y, x]
            if abs( disp_l[y][x] - disp_r[y][x - disp_l[y][x]] ) <= 1:  # correct
                validity_map[y][x] = 1
            else:
                for i in range(1, mismatch_range):
                    if x - i >= 0 and abs( disp_l[y][x - i] - disp_r[y][x - disp_l[y][x]] ) <= 1 :
                        validity_map[y][x] = 2 # mismatch
                        break
                    if x + i < w and abs( disp_l[y][x + i] - disp_r[y][x - disp_l[y][x]] ) <= 1 :
                        validity_map[y][x] = 2 # mismatch
                        break
                if validity_map[y][x] == 0:
                    validity_map[y][x] = 3 # occlusion                    

    ref_threshold = max_disp // 6
    for y in range(h): 
        for x in range(w):
            if validity_map[y][x] == 3:
                b = -1
                for i in range(1, x + 1):
                    if validity_map[y][x - i] == 1 and disp_l[y, x - i] > ref_threshold:
                        b = x - i
                        break
                
                a = -1
                for k in range(x + 1, w):
                    if validity_map[y, k] == 1 and disp_l[y, k] > ref_threshold:
                        a = k
                        break
                
                if a != -1 and b != -1:
                    if disp_l[y, a] < disp_l[y, b]:
                        disp[y, x] = disp_l[y, a]
                    else:
                        disp[y, x] = disp_l[y, b]
                elif a != -1:
                    disp[y, x] = disp_l[y, a]
                elif b != -1:
                    disp[y, x] = disp_l[y, b]
                
            elif validity_map[y][x] == 2:
                median = 0
                valid_cnt = 0
                for east in range(1, w - x): 
                    if validity_map[y][x + east] == 1:
                        median += disp_l[y][x + east]
                        valid_cnt += 1
                        break
                for south in range(1, h - y): 
                    if validity_map[y + south][x] == 1:
                        median += disp_l[y + south][x]
                        valid_cnt += 1
                        break
                for west in range(1, x + 1): 
                    if validity_map[y][x - west] == 1:
                        median += disp_l[y][x - west]
                        valid_cnt += 1
                        break
                for north in range(1, y + 1): 
                    if validity_map[y - north][x] == 1:
                        median += disp_l[y - north][x]
                        valid_cnt += 1
                        break
                for northeast in range( min(w - 1 - x, y) ): 
                    if validity_map[y - northeast][x + northeast] == 1:
                        median += disp_l[y - northeast][x + northeast]
                        valid_cnt += 1
                        break
                for southeast in range( min(w - 1 - x, h - 1 - y) ): 
                    if validity_map[y + southeast][x + southeast] == 1:
                        median += disp_l[y + southeast][x + southeast]
                        valid_cnt += 1
                        break
                for southwest in range( min(x, h - 1 - y) ): 
                    if validity_map[y + southwest][x - southwest] == 1:
                        median += disp_l[y + southwest][x - southwest]
                        valid_cnt += 1
                        break
                for northwest in range( min(x, y) ): 
                    if validity_map[y - northwest][x - northwest] == 1:
                        median += disp_l[y - northwest][x - northwest]
                        valid_cnt += 1
                        break
                disp[y][x] = int(median / valid_cnt)
    return disp

def computeDisp(Il, Ir, max_disp, scale_factor):
    global h, w, ch, Il_gray, Ir_gray, Il_lab, Ir_lab, dx1, dy1, dx2, dy2
    global costl, costr, phi_l, phi_r, labels_l, labels_r, valid, invalid
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.uint8)

    # >>> Cost computation
    tic = time.time()
    # TODO: Compute matching cost from Il and Ir
    Il_gray = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    Ir_gray = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)
    Il_lab = cv2.cvtColor(Il, cv2.COLOR_BGR2Lab).astype('float')
    Ir_lab = cv2.cvtColor(Ir, cv2.COLOR_BGR2Lab).astype('float')

    dx1 = np.around(np.random.normal(0, SIGMA, N)).astype('int')
    dy1 = np.around(np.random.normal(0, SIGMA, N)).astype('int')
    dx2 = np.around(np.random.normal(0, SIGMA, N)).astype('int')
    dy2 = np.around(np.random.normal(0, SIGMA, N)).astype('int')
    
    pool = mp.Pool()
    costl = []
    costr = []
    phi_l = []
    phi_r = []
    for tmp in pool.map(compute_cost, range(h)):
        costl.append(tmp[0])
        costr.append(tmp[1])
        phi_l.append(tmp[2])
        phi_r.append(tmp[3])
    costl = np.array(costl)
    costr = np.array(costr)
    phi_l = np.array(phi_l)
    phi_r = np.array(phi_r)

    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

    # >>> Cost aggregation
    tic = time.time()
    # TODO: Refine cost by aggregate nearby costs
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

    # >>> Disparity optimization
    tic = time.time()
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    pool = mp.Pool()
    labels_l = []
    labels_r = []
    dis_l = []
    for tmp in pool.map(find_disp, range(h)):
        labels_l.append(tmp[0])
        labels_r.append(tmp[1])
        dis_l.append(tmp[2])
    labels_l = np.array(labels_l)
    labels_r = np.array(labels_r)
    dis_l = np.array(dis_l)

    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
    tic = time.time()
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    # labels = check_validity(labels_l.astype(np.int), labels_r.astype(np.int))
    invalid = []
    valid = np.zeros((h, w), dtype=np.bool)
    print(np.sum(labels_l[h//30: h - h//30, w//10: w - w//10] < pad), (h - (h//30)*2) * (w - (w//10)*2) / 7)
    if np.sum(labels_l[h//30: h - h//30, w//10: w - w//10] < pad) < (h - (h//30)*2) * (w - (w//10)*2) / 7:
        for i in range(h):
            for j in range(w):
                d = int(labels_l[i, j])
                if np.abs(labels_l[i, j] - labels_r[i, j - d]) <= 1 and d >= pad:
                    valid[i, j] = True
                else:
                    invalid.append((i, j))
    else:
        for i in range(h):
            for j in range(w):
                d = int(labels_l[i, j])
                if np.abs(labels_l[i, j] - labels_r[i, j - d]) <= 1 and d <= max_disp - pad and d >= pad // 6:
                    valid[i, j] = True
                else:
                    invalid.append((i, j))
    
    labels = labels_l.copy()
    if np.sum(labels_l[h//30: h - h//30, w//10: w - w//10] < pad) < (h - (h//30)*2) * (w - (w//10)*2) / 7:
        ref_threshold = max(np.mean(labels) - np.std(labels) * 2, max_disp // 6 + pad)
    else:
        ref_threshold = max(np.mean(labels) - np.std(labels) * 2, max_disp // 6)
    for i, j in invalid:
        a = -1
        for k in range(j + 1, w):
            if valid[i, k] == True and labels_l[i, k] > ref_threshold:
                a = k
                break
        b = -1
        for k in range(j - 1, -1, -1):
            if valid[i, k] == True and labels_l[i, k] > ref_threshold:
                b = k
                break
        if a != -1 and b != -1:
            if labels_l[i, a] < labels_l[i, b]:
                labels[i, j] = labels_l[i, a]
            else:
                labels[i, j] = labels_l[i, b]
        elif a != -1:
            labels[i, j] = labels_l[i, a]
        elif b != -1:
            labels[i, j] = labels_l[i, b]
        elif labels_l[i, j] < pad:
            labels[i, j] = np.mean(labels[
                max(i - 10, 0): min(i + 10, h - 1),
                max(j - 10, 0): min(j + 10, w - 1)])
    
    labels = cv2.medianBlur(np.uint8(labels * scale_factor), 5)
    labels = cv2.bilateralFilter(labels, 5, 9, 16)

    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))

    return labels

def main():
    global max_disp
    args = parser.parse_args()

    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    if len(img_left.shape) == 3:
        img_left = np.pad(img_left, ((0, 0), (pad, 0), (0, 0)), 'constant', constant_values=255)
        img_right = np.pad(img_right, ((0, 0), (0, pad), (0, 0)), 'constant', constant_values=255)
    else:
        img_left = np.pad(img_left, ((0, 0), (pad, 0)), 'constant', constant_values=255)
        img_right = np.pad(img_right, ((0, 0), (0, pad)), 'constant', constant_values=255)
    tic = time.time()
    max_disp = 80
    scale_factor = 1
    disp = computeDisp(img_left, img_right, max_disp, scale_factor)
    disp = disp[: , pad: ]
    if np.sum(disp < pad) < h * w / 7:
        disp = np.clip(disp, pad, max_disp)
        disp -= 20
    toc = time.time()
    writePFM(args.output, disp.astype(np.float32))
    print('Elapsed time: %f sec.' % (toc - tic))

if __name__ == '__main__':
    main()
