from garl.config import Config
import garl.main_utils as utils
import os
import joblib
from garl.coinrunenv import init_args_and_threads


def load_for_setup_if_necessary():
    restore_file(Config.RESTORE_ID)

def restore_file(restore_id,base_name=None,overlap_config=None,load_key='default'):
    """overlap config means you can modify the config in savefile, e.g. test seed"""
    if restore_id is not None:
        load_file = Config.get_load_filename(restore_id=restore_id,base_name=base_name)
        filepath = file_to_path(load_file)
        assert os.path.exists(filepath),"don't exist"
        load_data = joblib.load(filepath)

        Config.set_load_data(load_data, load_key=load_key)

        restored_args = load_data['args']
        sub_dict = {}
        res_keys = Config.RES_KEYS

        for key in res_keys:
            if key in restored_args:
                sub_dict[key] = restored_args[key]
            else:
                print('warning key %s not restored' % key)

        Config.parse_args_dict(sub_dict)
        if overlap_config is not None:
            Config.parse_args_dict(overlap_config)

    print(Config.SET_SEED,Config.NUM_LEVELS)
    print("Init coinrun env threads and env args")
    init_args_and_threads(4)
    if restore_id == None:
        return None
    else:
        return load_file

# push loaddata['args'] into config
def restore_file_back(restore_id, load_key='default'):
    if restore_id is not None:
        load_file = Config.get_load_filename(restore_id=restore_id)
        filepath = file_to_path(load_file)
        load_data = joblib.load(filepath)

        Config.set_load_data(load_data, load_key=load_key)

        restored_args = load_data['args']
        sub_dict = {}
        res_keys = Config.RES_KEYS

        for key in res_keys:
            if key in restored_args:
                sub_dict[key] = restored_args[key]
            else:
                print('warning key %s not restored' % key)

        Config.parse_args_dict(sub_dict)

    from coinrun.coinrunenv import init_args_and_threads
    init_args_and_threads(4)

# push loaddata['args'] into config
def restore_checkpoint(restore_id, checkpoint=32, load_key='default'):
    if restore_id is not None:
        load_file = Config.get_load_filename(base_name=str(checkpoint)+'M',restore_id=restore_id)
        #load_file = Config.get_load_filename(restore_id=restore_id)
        filepath = file_to_path(load_file)
        load_data = joblib.load(filepath)

        Config.set_load_data(load_data, load_key=load_key)

        restored_args = load_data['args']
        sub_dict = {}
        res_keys = Config.RES_KEYS

        for key in res_keys:
            if key in restored_args:
                sub_dict[key] = restored_args[key]
            else:
                print('warning key %s not restored' % key)

        Config.parse_args_dict(sub_dict)

    from coinrun.coinrunenv import init_args_and_threads
    init_args_and_threads(4)


# setup at first
def setup_and_load(use_cmd_line_args=True, **kwargs):
    """
    Initialize the global config using command line options, defaulting to the values in `config.py`.

    `use_cmd_line_args`: set to False to ignore command line arguments passed to the program
    `**kwargs`: override the defaults from `config.py` with these values
    """
    args = Config.initialize_args(use_cmd_line_args=use_cmd_line_args, **kwargs)

    #load_for_setup_if_necessary()

    return args

def file_to_path(filename):
    return os.path.join(Config.LOGDIR, filename)
