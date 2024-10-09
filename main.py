from Shape import *
from multiprocessing import Pool

def example(optimization):
    env = Enviroment()
    env.build_env()
    env.draw(path=r'C:\Technion\RL\Code\Basic problem\figs\steering optimization',optimized=None)

    env.Adam_optimize(optimized=optimization,learning_rate=60)
    env.draw(path=r'C:\Technion\RL\Code\Basic problem\figs\steering optimization',optimized=optimization,mode='Adam')

    env.optimize(optimized=optimization)
    env.draw(path=r'C:\Technion\RL\Code\Basic problem\figs\steering optimization',optimized=optimization)

    env.Adam_optimize(optimized=optimization)
    env.draw(path=r'C:\Technion\RL\Code\Basic problem\figs\steering optimization',optimized=optimization,mode='Combined')

def scenario_optimization(optimization):
    #Building and optimizing scenario
    env = Enviroment()
    env.build_env()
    init_coverage = env.compute_coverage()
    # print(f'In function init_coverage :{init_coverage}')
    env.Adam_optimize()
    adam_coverage = env.compute_coverage()
    env.optimize(optimized=optimization)
    bf_coverage = env.compute_coverage()
    env.Adam_optimize()
    combined_coverage = env.compute_coverage()
    return [init_coverage, adam_coverage, bf_coverage, combined_coverage]

def init(device):
    global shared_device
    shared_device = device


def stats(optimization):
    init_coverage = []
    adam_coverage= []
    bf_coverage = []
    combined_coverage = []


    # df_stats = pd.read_csv(os.path.join(r'C:\Technion\RL\Code\Basic problem\data tables','stats.csv'))
    with Pool(processes=2) as pool:  # Number of processes
        results = pool.starmap(scenario_optimization, [(optimization,) for _ in range(2)])


    init_coverage = [x[0] for x in results]
    print(f'First init: {init_coverage}')
    adam_coverage = [x[1] for x in results]
    print(f'First adam_coverage: {adam_coverage}')
    bf_coverage = [x[2] for x in results]
    combined_coverage = [x[3] for x in results]
            


    data={'init_coverage':init_coverage,
          'adam_coverage':adam_coverage,
          'bf_coverage':bf_coverage,
          'combined_coverage':combined_coverage}
    df_stats = pd.DataFrame(data=data)
    df_stats['Ratio_adam'] = df_stats['adam_coverage'] / df_stats['init_coverage']
    df_stats['Ratio_bf'] = df_stats['bf_coverage'] / df_stats['init_coverage']
    df_stats['Ratio_combined'] = df_stats['combined_coverage'] / df_stats['init_coverage']
    df_stats['Gap_adam'] = df_stats['adam_coverage'] - df_stats['init_coverage']
    df_stats['Gap_bf'] = df_stats['bf_coverage'] - df_stats['init_coverage']
    df_stats['Gap_combined'] = df_stats['combined_coverage'] - df_stats['init_coverage']
    df_stats.to_csv(os.path.join(r'C:\Technion\RL\Code\Basic problem\data tables','stats.csv'),mode='a',
    index=False,header=False)
    # df_melted = df_stats.melt(value_vars=['Ratio_2_to_1', 'Ratio_3_to_1', 'Ratio_4_to_1'], var_name='Ratio Type', value_name='Ratio')
    fig, ax = plt.subplots(figsize=(10, 6))
    for opt_name in ['adam','bf','combined']:
        ax.hist(df_stats[f'Ratio_{opt_name}'],label=f'{opt_name} Ratio',alpha=0.3)
    ax.legend()
    # Plot histograms
    
    # sns.histplot(data=df_melted,ax=ax, x='Ratio', hue='Ratio Type', bins=10, alpha=0.5)
    ax.set_xlabel('Ratio')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Ratios')
    plt.show()    



if __name__ == "__main__":

    optimization='steering'
    # stats(optimization='steering')
    example(optimization=optimization)

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
