import os
import numpy as np
from Parameters import check_path
from Visualize import get_plt


exp = 0     # 第几套实验
num_epochs = 100    # 重复实验次数
results_root = 'results/exp{}'.format(exp)
pwt_results_path = os.path.join(results_root, 'raw_pwt_curves')
schedule_path = os.path.join(results_root, 'raw_optimized_schedules')
processed_path = os.path.join(results_root, 'processed_results')
check_path(processed_path)

def save_processed_results():
    np.save(os.path.join(processed_path, 'mean_pwt.npy'), mean_pwts)
    with open(os.path.join(processed_path, 'statistic_results.txt'),'w') as f:
        msg = 'PWT Results' + '\n'
        msg += 'num_generations: {}'.format(num_generations) + '\n'
        msg += 'point_step: {}'.format(point_step) + '\n'
        msg += 'check_points: {}'.format(check_points) + '\n'
        msg += 'mean_pwts: {}'.format(mean_pwt) + '\n'
        msg += 'relative_mean_pwts: {}'.format(relative_mean_pwt) + '\n'
        msg += 'std_pwts: {}'.format(std_pwts) + '\n'
        msg += 'relative_std_pwts: {}'.format(relative_std_pwts) + '\n'

        #msg += '\nSchedule Results' + '\n'
        #msg += ': {}'.format() + '\n'
        f.write(msg)
    f.close()

num_generations = 800
point_step = 100
num_points = num_generations // point_step
check_points = [point_step * (i + 1) for i in range(num_points)]

mean_pwts = []
mean_pwt = []
relative_mean_pwt = []
std_pwts = [[] for _ in range(num_points)]
relative_std_pwts = []

def process_pwt():
    global mean_pwts
    for epoch in range(num_epochs):
        min_pwt_epoch = np.load(os.path.join(pwt_results_path, 'pwt_epoch{}.npy'.format(epoch)))
        if epoch == 0:
            mean_pwts = min_pwt_epoch
        else:
            mean_pwts += min_pwt_epoch
        for index, point in enumerate(check_points):
            std_pwts[index].append(min_pwt_epoch[point])
    mean_pwts = mean_pwts / num_epochs
    for index in range(num_points):
        mean_pwt.append(round(mean_pwts[check_points[index]] * 100) / 100)
        relative_mean_pwt.append(round(mean_pwt[index] / mean_pwts[0] * 10000) / 100)
        std_pwts[index] = round(np.std(std_pwts[index]) * 100) / 100
        relative_std_pwts.append(round(std_pwts[index] / mean_pwt[index] * 10000) / 100)

process_pwt()

print('Processing data of exp{}...'.format(exp))

print('\nmean_pwts:')
print(mean_pwt)

print('\nrelative_mean_pwt:')
print(relative_mean_pwt)

print('\nstd_pwts:')
print(std_pwts)

print('\nrelative_std_pwts:')
print(relative_std_pwts)

save_processed_results()

plt = get_plt('Mean PWT', mean_pwts)
plt.savefig(os.path.join(processed_path, 'mean_pwt.png'))
plt.show()

