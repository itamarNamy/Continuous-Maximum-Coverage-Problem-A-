from Shape import *
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import datetime
import os
from itertools import product

def drawing_example(optimization,path,save = False):
    fig, ax = plt.subplots(figsize=(14, 14),nrows=2,ncols=2)

    env = Enviroment()
    env.draw(ax[0][0], optimized=None)

    env.Adam_optimize(optimized=optimization)
    env.draw(ax[0][1], optimized=optimization,mode='Adam')

    env.optimize(optimized=optimization,angles_num = 10,x_coords_num = 8,y_coords_num = 8)
    env.draw(ax[1][0], optimized=optimization)
    
    env.Adam_optimize(optimized=optimization)
    env.draw(ax[1][1], optimized=optimization,mode='Combined')
    now = datetime.datetime.now()
    if save:
        fig.savefig(os.path.join(path,now.strftime("%m%d-%H%M") +'.png')  ,bbox_inches='tight')
    plt.show()

def greedy_combined_comp(optimization,env_args):
    stats_keys = ['init_coverage', 'adam_coverage', 'bf_coverage', 'combined_coverage', 'systems_area',
                   'protection_area','Number_of_systems','Number_of_areas','avreage_system_size',
            'average_area_size','precision','recall','Fscore']
    run_stats = dict.fromkeys(stats_keys)
    #Building and optimizing scenario
    env = Enviroment(**env_args)

    run_stats['init_coverage'] = env.compute_coverage()
    run_stats['systems_area'] = env.systems_area
    run_stats['protection_area'] = env.areas.unitedArea.area
    run_stats['env_area'] = np.linalg.norm(env.env_borders)
    env.Adam_optimize()
    run_stats['adam_coverage'] = env.compute_coverage()
    env.optimize(optimized=optimization)
    run_stats['bf_coverage'] = env.compute_coverage()
    env.Adam_optimize()
    run_stats['combined_coverage'] = env.compute_coverage()
    run_stats['Number_of_systems'] = env.n_systems
    run_stats['Number_of_areas'] = env.n_polygons
    run_stats['average_system_size'] = run_stats['systems_area'] / run_stats['Number_of_systems']
    run_stats['average_area_size'] = run_stats['protection_area'] / run_stats['Number_of_areas']
    run_stats['precision'] = run_stats['combined_coverage']/run_stats['systems_area']
    run_stats['recall'] = run_stats['combined_coverage']/run_stats['protection_area']
    run_stats['Fscore'] = run_stats['recall'] * run_stats['precision'] / (run_stats['recall'] + run_stats['precision'])
    return run_stats

def run_optimization(env:Enviroment, method:str, opt_kind:str ,params:dict)->float:
    if method == 'Adam':
        env.Adam_optimize(delta = params['delta'])
    elif method == 'Combined':
        env.optimize(optimized = opt_kind)
        env.Adam_optimize(delta = params['delta'])
    return env.compute_coverage()

def params_opt():
    env = Enviroment()
    protection_area = env.areas.unitedArea.area
    env.optimize(optimized='location')
    deltas = protection_area * np.logspace(-7,1,num=9)
    params = {}
    deltas_stats = []
    for delta in deltas:
        curr_env = env.copy()
        params['delta'] = delta
        coverage = run_optimization(curr_env, method='Adam', opt_kind='location', params = params)
        print(f'delta {delta/protection_area:.6f}:{coverage}')
        deltas_stats.append({'delta':delta/protection_area,'coverage':coverage})
    deltas_df = pd.DataFrame(deltas_stats)
    output_path = os.path.join(r'C:\Technion\RL\Code\Basic problem\data tables','deltas stats.csv')
    deltas_df.to_csv(output_path,mode='a',index=False,header=not os.path.exists(output_path))




def stats(optimization, repeats = 2):
    border_vals = np.linspace(5, 15, 2)  # [5.0, 10.0, 15.0]
    borders = list(((-v, v), (-v, v)) for v in border_vals)
    env_params_values = {'n_systems': np.arange(1,2,1), 'n_polygons': np.arange(99,100,1),
                         'polygon_size':np.linspace(0.5,2,2),'env_borders':borders}
    keys = list(env_params_values.keys())
    value_lists = list(env_params_values.values())
    
    # Compute the product of all value lists
    combos = product(*value_lists)
    
    # For each tuple in the product, zip it with the keys to form a dictionary
    dict_combs = []
    for c in combos:
        combination_dict = dict(zip(keys, c))
        dict_combs.append(combination_dict)
    results = []
    with ProcessPoolExecutor() as executor:

        futures = [executor.submit(greedy_combined_comp,optimization, dict_combs[i]) for i in range(len(dict_combs))
                   for _ in range(repeats)]

        # Process the results as they complete
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            # stats_dict = {key: None for key in keys}
            
            # curr_results = future.result()

            # for key_i,key in enumerate(keys[:-3]):#precision and recall can be computed only after loop
            #     stats_dict[key] = curr_results[key_i]

            
    
    results_df = pd.DataFrame(results)
    output_path = os.path.join(r'.\data tables',optimization + ' stats.csv')
    results_df.to_csv(output_path,mode='a',index=False,header=not os.path.exists(output_path))


    # df_melted = df_stats.melt(value_vars=['Ratio_2_to_1', 'Ratio_3_to_1', 'Ratio_4_to_1'], var_name='Ratio Type', value_name='Ratio')
    # fig, ax = plt.subplots(figsize=(10, 6))
    # for opt_name in ['adam','bf','combined']:
    #     ax.hist(df_stats[f'Ratio_{opt_name}'],label=f'{opt_name} Ratio',alpha=0.3)
    # ax.legend()
    # Plot histograms
    
    # sns.histplot(data=df_melted,ax=ax, x='Ratio', hue='Ratio Type', bins=10, alpha=0.5)
    # ax.set_xlabel('Ratio')
    # ax.set_ylabel('Frequency')
    # ax.set_title('Histogram of Ratios')
    # plt.show()    



if __name__ == "__main__":
    optimization='steering'
    stats(optimization)
    # location / steering
    # path = r'.\figs\\' + optimization + ' optimization'
    # stats(optimization=optimization)
    # drawing_example(optimization=optimization,path = path, save= True)
    # params_opt()

    # running_times = []
    # for _ in range(2):
    #     env = Enviroment()
    #     env.build_env()
    #     start = time.time()
    #     env.optimize()
    #     stop = time.time()
    #     running_times.append(stop-start)
    # print(f'runnig time: {running_times}')
    # os.chdir(r'C:\Technion\RL\Code\Basic problem\running times')

    # with open('phase_2_running_times.txt', "w") as file:
    #      file.write(str(running_times))
    # env.draw(path=r'C:\Technion\RL\Code\Basic problem\figs',optimized=True)
