import os
import logging
import multiprocessing as mp

import pprint
import yaml
import torch

from src.train import main as app_main_mvtec
from src.AD import main as AD, _demo

import hydra
from omegaconf import DictConfig, OmegaConf


def process_main(rank: int, cfg_dict: dict, world_size: int):
    """
    rank: local rank
    cfg_dict: dict config
    world_size: len(devices)
    """
    # ----
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)

    devices = cfg_dict.get("devices", ["cuda:0"])
    mode = cfg_dict.get("mode", "train")
    dist = cfg_dict.get("dist", {})
    master_addr = dist.get("master_addr", "localhost")
    master_port = str(dist.get("master_port", 40112))
    import platform
    backend = dist.get("backend", "nccl")
    if platform.system() == "Windows":
        backend = "gloo"

    dev_str = devices[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_str.split(":")[-1])

    params = cfg_dict.get("app", {}) # model config

    params.update(cfg_dict)

    logger.info("Params:")
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(params)

    # ---- DDP    
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank
    )

    if mode == "train":
        if rank==0:
            log_dir = params["logging"]["folder"]
            os.makedirs(log_dir, exist_ok=True)
            params_save_path = os.path.join(log_dir, "params.yaml")
            with open(params_save_path, "w") as f:
                yaml.safe_dump(params, f, default_flow_style=False, sort_keys=False)
            print(f"Config saved to {params_save_path}")

        app_main_mvtec(args=params)
    elif mode == "AD":
        run_name = cfg_dict.get("run_name")
        variant = cfg_dict.get("variant", "Ours")
        data_name = cfg_dict['data']['data_name']
        
        if run_name:
            load_path = os.path.join('logs', run_name, variant, data_name)
        else:
            load_path = os.path.join('logs', data_name, params.get('model_name','')+cfg_dict['diy_name'])
            
        saved_path = os.path.join(load_path, "params.yaml")
        if os.path.exists(saved_path):
            with open(saved_path, "r") as f:
                saved_params = yaml.safe_load(f)
            
            # Merge saved meta into current params
            params['meta'] = saved_params.get('meta', {})
            
            # Set checkpoint path
            ckpt_epoch = params.get('ckpt_epoch', 50)
            params["ckpt_path"] = os.path.join(load_path, f"train-epoch{ckpt_epoch}.pth.tar")
            params["logging"]["folder"] = os.path.join(load_path, f"eval/{str(ckpt_epoch)}")
            
            print(f"🚀 [AD Mode] Loading weights from: {params['ckpt_path']}")
            AD(args=params)
        else:
            print(f"❌ No params.yaml found at {saved_path}")
    elif mode == "demo":
        run_name = cfg_dict.get("run_name")
        variant = cfg_dict.get("variant", "Ours")
        data_name = cfg_dict['data']['data_name']
        
        if run_name:
            load_path = os.path.join('logs', run_name, variant, data_name)
        else:
            load_path = os.path.join('logs', data_name, params.get('model_name','')+cfg_dict['diy_name'])
            
        saved_path = os.path.join(load_path, "params.yaml")
        if os.path.exists(saved_path):
            with open(saved_path, "r") as f:
                saved_params = yaml.safe_load(f)
            
            params['meta'] = saved_params.get('meta', {})
            params["ckpt_path"] = os.path.join(load_path, f"train-epoch50.pth.tar") # Default to 50
            params["logging"]["folder"] = os.path.join(load_path, "demo")
            
            print(f"📺 [Demo Mode] Loading {params['ckpt_path']}...")
            _demo(params["ckpt_path"], params)
        else:
            print(f"❌ No ckpt found at {saved_path}")
    else:
        if rank == 0:
            logger.error(f"Unknown mode: {mode}")

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    mode = cfg_dict.get("mode", "train")
    if mode == "train":
        import datetime
        now = datetime.datetime.now()
        timestamp = now.strftime("%Hh%M_%d-%m-%Y")
        
        data_name = cfg_dict.get("data", {}).get("data_name", "dataset")
        model_name = cfg_dict.get("app", {}).get("meta", {}).get("model", "model")
        
        # New structure logic
        run_name = cfg_dict.get("run_name")
        if run_name is None:
            base_logs = "logs"
            run_idx = 1
            if os.path.exists(base_logs):
                existing = [d for d in os.listdir(base_logs) if d.startswith(model_name)]
                run_idx = len(existing) + 1
            run_name = f"{model_name}_{run_idx}-{timestamp}"
        
        variant = cfg_dict.get("variant", "default")
        
        # logs/dinov3_1-18h06.../Ours/bottle
        if "logging" not in cfg_dict["app"]:
            cfg_dict["app"]["logging"] = {}
        cfg_dict["app"]["logging"]["folder"] = os.path.join("logs", run_name, variant, data_name)

    devices = cfg_dict.get("devices", ["cuda:0"])
    world_size = len(devices)

    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(world_size):
        p = mp.Process(target=process_main, args=(rank, cfg_dict, world_size))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

if __name__ == '__main__':
    main()