from Module import *
from Visualize import visual_vals
from numpy.random import shuffle



def GA(initial_populations, num_generations, num_services, num_stations, transfer_Matrix, 
        all_ODMatrix, time_from_start, reachable_stations, time_to_transfer, time_span, epoch, num_epochs):
    min_PWTs = []
    best_p = []
    cost_time = 0
    father_populations = initial_populations

    father_vals = [0 for i in range(len(father_populations))]
    t_start = time.time()
    process_pool = Pool(os.cpu_count())
    def change_father_vals(args):
        father_vals[args[0]] = args[1]
    for val_pos, p in enumerate(father_populations):
        # 使用速度更快的计算策略
        process_pool.apply_async(calculate_pwt, (p, num_services, num_stations, all_ODMatrix, transfer_Matrix, time_from_start, 
                                reachable_stations, time_to_transfer, time_span, 
                                train_capacity, start_station_index, val_pos), callback=change_father_vals)
    process_pool.close()
    process_pool.join()

    min_PWTs.append(min(father_vals))
    for generation in range(num_generations):
        t_stop = time.time()
        gen_time = t_stop - t_start
        cost_time += gen_time
        print("epoch {}/{}  |  gen {}/{}: {} / {}  |  time: {:.4f} / {:.4f}".format(epoch, num_epochs - 1, generation, num_generations, 
                                                    min_PWTs[-1], min_PWTs[0], gen_time, cost_time))
        t_start = time.time()
        cross_epsilon, mutation_epsilon = get_epsilon(generation)
        father_index_1 = select_tournament(father_vals, 2)
        father_index_2 = select_tournament(father_vals, 2)
        father_1 = father_populations[father_index_1].copy()
        father_2 = father_populations[father_index_2].copy()
        if rand() < cross_epsilon:
            child_1, child_2 = cross_by_line(father_1, father_2)
        else:
            child_1, child_2 = father_1, father_2
        child_1 = mutation(child_1, mutation_epsilon)
        child_2 = mutation(child_2, mutation_epsilon)
        child_populations = np.vstack((child_1, child_2))
        while len(child_populations) < num_populations:
            father_index_1 = select_tournament(father_vals, 2)
            father_index_2 = select_tournament(father_vals, 2)
            father_1 = father_populations[father_index_1].copy()
            father_2 = father_populations[father_index_2].copy()
            if rand() < cross_epsilon:
                child_1, child_2 = cross_by_line(father_1, father_2)
            else:
                child_1, child_2 = father_1, father_2
            child_1 = mutation(child_1, mutation_epsilon)
            child_2 = mutation(child_2, mutation_epsilon)
            child_populations = np.vstack((child_populations, child_1, child_2))
        # 选择下一代

        child_vals = [0 for i in range(len(child_populations))]

        process_pool = Pool(os.cpu_count())
        def change_child_vals(args):
            child_vals[args[0]] = args[1]
        for val_pos, p in enumerate(child_populations):
            # 使用速度更快的计算策略
            process_pool.apply_async(calculate_pwt, (p, num_services, num_stations, all_ODMatrix, transfer_Matrix, time_from_start, 
                                    reachable_stations, time_to_transfer, time_span, train_capacity, 
                                    start_station_index, val_pos), callback=change_child_vals)
        process_pool.close()
        process_pool.join()

        all_populations = np.vstack((father_populations, child_populations))
        all_vals = father_vals + child_vals
        vals_copy = all_vals.copy()
        vals_sorted= list(set(vals_copy))
        vals_sorted.sort()
        all_vals = np.array(all_vals)
        next_populations = []
        next_vals = []
        i = 0
        min_PWTs.append(vals_sorted[0])
        while len(next_populations) < num_populations:
            pop_i = all_populations[all_vals == vals_sorted[i]]
            pop_i = np.unique(pop_i, axis=0)
            if i == 0:
                best_p = pop_i[0].copy()
            if len(pop_i) + len(next_populations) <= num_populations:
                if len(next_populations) == 0:
                    next_populations = pop_i
                else:
                    next_populations = np.vstack((next_populations, pop_i))
                next_vals = next_vals + [vals_sorted[i]] * len(pop_i)
            else:
                lack_num = num_populations - len(next_populations)
                if len(next_populations) == 0:
                    next_populations = pop_i[:lack_num]
                else:
                    next_populations = np.vstack((next_populations, pop_i[:lack_num]))
                next_vals = next_vals + [vals_sorted[i]] * lack_num
            i += 1
        next_vals = np.array(next_vals)
        next_data = np.column_stack((next_populations, next_vals))
        shuffle(next_data)
        father_populations = np.array(next_data[:, :-1], dtype=int)
        father_vals = list(next_data[:, -1])
    t_stop = time.time()
    gen_time = t_stop - t_start
    cost_time += gen_time
    print("epoch {}/{}  |  gen {}/{}: {} / {}  |  time: {:.4f} / {:.4f}".format(epoch, num_epochs - 1, num_generations, num_generations, 
                                                    min_PWTs[-1], min_PWTs[0], gen_time, cost_time))
    return best_p, min_PWTs, cost_time

if __name__ == "__main__":
    exp = 0     # 第几套实验
    results_path = './results/exp{}'.format(exp)
    pwt_results_path = os.path.join(results_path, 'raw_pwt_curves')
    schedule_path = os.path.join(results_path, 'raw_optimized_schedules')
    
    check_path(pwt_results_path)
    check_path(schedule_path)
    
    save_config(results_path)

    cost_time = 0
    num_epochs = 100
    for epoch in range(num_epochs):
        try:
            print('start epoch {}/{}'.format(epoch, num_epochs - 1))
            best_p, min_PWTs, epoch_time = GA(initial_populations, num_generations, num_services, num_stations, transfer_Matrix,
                            all_ODMatrix, time_from_start, reachable_stations, time_to_transfer, time_span, epoch, num_epochs)
            
            np.save(os.path.join(pwt_results_path, 'pwt_epoch{}.npy'.format(epoch)), min_PWTs)
            np.save(os.path.join(schedule_path, 'schedule_epoch{}.npy'.format(epoch)), best_p)
            cost_time += epoch_time
            print('finish epoch {}/{}  |  time: {:.4f} / {:.4f}\n'.format(epoch, num_epochs - 1, epoch_time, cost_time))
        except Exception as e:
            print(e)
            exit(0)