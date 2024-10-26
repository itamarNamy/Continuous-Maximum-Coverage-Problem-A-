from Shape import *
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

def example(optimization,path,save = False):
    env = Enviroment()
    env.draw(path, save, optimized=None)

    env.Adam_optimize(optimized=optimization)
    env.draw(path, save, optimized=optimization,mode='Adam')

    env.optimize(optimized=optimization)
    env.draw(path, save, optimized=optimization)

    env.Adam_optimize(optimized=optimization)
    env.draw(path, save, optimized=optimization,mode='Adam')

def scenario_optimization(optimization):
    #Building and optimizing scenario
    env = Enviroment()
    init_coverage = env.compute_coverage()
    systems_area = env.systems_area
    protection_area = env.areas.unitedArea.area
    # print(f'In function init_coverage :{init_coverage}')
    env.Adam_optimize()
    adam_coverage = env.compute_coverage()
    env.optimize(optimized=optimization)
    bf_coverage = env.compute_coverage()
    env.Adam_optimize()
    combined_coverage = env.compute_coverage()
    return [init_coverage, adam_coverage, bf_coverage, combined_coverage, systems_area, protection_area]



def stats(optimization):

    results = []


    # df_stats = pd.read_csv(os.path.join(r'C:\Technion\RL\Code\Basic problem\data tables','stats.csv'))
    # with Pool(processes=2) as pool:  # Number of processes
    
    #     results = pool.starmap(scenario_optimization, [(optimization,) for _ in range(2)])
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(scenario_optimization,optimization) for _ in range(4)]

        # Process the results as they complete
        for future in as_completed(futures):
            
            keys = ['init_coverage', 'adam_coverage', 'bf_coverage', 'combined_coverage', 'systems_area', 'protection_area',
            'precision','recall','Fscore']
            stats_dict = {key: None for key in keys}
            
            curr_results = future.result()

            for key_i,key in enumerate(keys[:-3]):#precision and recall can be computed only after loop
                stats_dict[key] = curr_results[key_i]
            stats_dict['precision'] = stats_dict['combined_coverage']/stats_dict['systems_area']
            stats_dict['recall'] = stats_dict['combined_coverage']/stats_dict['protection_area']
            stats_dict['Fscore'] = stats_dict['recall'] * stats_dict['precision'] / (stats_dict['recall'] + stats_dict['precision'])
            results.append(stats_dict)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(r'C:\Technion\RL\Code\Basic problem\data tables','stats.csv'),mode='a',
    index=False)



    # data={'init_coverage':init_coverage,
    #       'adam_coverage':adam_coverage,
    #       'bf_coverage':bf_coverage,
    #       'combined_coverage':combined_coverage}
    # df_stats = pd.DataFrame(data=data)
    # df_stats['Ratio_adam'] = df_stats['adam_coverage'] / df_stats['init_coverage']
    # df_stats['Ratio_bf'] = df_stats['bf_coverage'] / df_stats['init_coverage']
    # df_stats['Ratio_combined'] = df_stats['combined_coverage'] / df_stats['init_coverage']
    # df_stats['Gap_adam'] = df_stats['adam_coverage'] - df_stats['init_coverage']
    # df_stats['Gap_bf'] = df_stats['bf_coverage'] - df_stats['init_coverage']
    # df_stats['Gap_combined'] = df_stats['combined_coverage'] - df_stats['init_coverage']
    # df_stats.to_csv(os.path.join(r'C:\Technion\RL\Code\Basic problem\data tables','stats.csv'),mode='a',
    # index=False,header=False)
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
        # location / steering
    path = r'C:\Technion\RL\Code\Basic problem\figs\\' + optimization + ' optimization'
    stats(optimization='steering')
    # example(optimization=optimization,path = path, save= True)

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
