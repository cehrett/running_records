import wandb
import pandas as pd
import logging
from math import gcd


best_runs = {
    'SUB': 'witw/running_records/gw65ptkg',
    'REP': 'witw/running_records/okanueqm',
    'DEL': 'witw/running_records/g13diptu'
}

varying_params = {
    'enc_heads': range(2, 13, 2),
    'enc_layers': range(2, 9, 1),
    'dec_heads': range(2, 13, 2),
    'dec_layers': range(2, 9, 1),
    'hid_dim_nheads_multiplier': range(6, 51, 4),
    'bpe_vocab_size': range(400, 3200, 200)
}

def lcm(a, b):
    return abs(a*b) // gcd(a, b)

def generate_runs(original_run, params_to_vary, model_type: str):
    original_config = original_run.config
    new_runs = []
    for param in params_to_vary:
        print("Varying param: ", param)
        print("Expected new runs basd on param: ", len(params_to_vary[param]))
        for value in params_to_vary[param]:
            new_config = original_config.copy()
            if "heads" not in param:
                new_config[param] = value
            else:
                # We have to adjust the hid_dim_nheads_multiplier to keep the hidden dimension same
                new_config[param] = value
                new_config['hid_dim_nheads_multiplier'] = new_config['hid_dim'] // lcm(new_config['enc_heads'], new_config['dec_heads'])
            new_config['model_type'] = model_type
            new_config['varied_param'] = param
            new_runs.append(new_config)

    return new_runs

if __name__ == "__main__":
    # Print the total number of new runs that will be generated based on the
    # varying parameters
    total_runs = 0
    for param in varying_params:
        total_runs += len(varying_params[param]) 
    print(f'Total number of new runs: {total_runs}')

    # api_instance = wandb.Api()
    # new_runs_df = pd.DataFrame()
    # for model_type, run_id in best_runs.items():
    #     print(f'Generating runs for {model_type} model')
    #     run = api_instance.run(run_id)
    #     print(f'Best run config: {run.config}')
    #     new_runs = generate_runs(run, varying_params, model_type)
    #     print(f'Generated {len(new_runs)} runs for {model_type} model')
    #     new_runs_df = new_runs_df.append(new_runs, ignore_index=True)

    # new_runs_df.to_csv('new_runs.csv', index=False)
    # exit(0)