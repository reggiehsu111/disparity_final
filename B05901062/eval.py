from util import readPFM, cal_avgerr

avg = 0
for i in range(10):
    gt = readPFM('./data/Synthetic/TLD{}.pfm'.format(i))
    disp = readPFM('./output/spad/TLD{}.pfm'.format(i))
    # disp = readPFM('./data/Synthetic/TLD0.pfm')
    print(disp.shape)
    avg += cal_avgerr(gt, disp)
    print('TLD{} avg err: '.format(i), cal_avgerr(gt, disp))
print('tot avg err: ', avg / 10)
