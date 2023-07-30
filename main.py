import fire
import pprint
import os
import wandb 

from data_loader import DataLoader
from model_loader import NeoXLoraModelLoader
from utils import tprint, Hparam


HP_ROOT_PATH = os.path.dirname(__file__)
hp = Hparam(HP_ROOT_PATH)
pp = pprint.PrettyPrinter(indent=4, width=1000)
# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="tabby-homework"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:734"


def finetune(hparam = 'finetune.yaml', **kwargs):
    hp.set_hparam(hparam, kwargs)

    dataloader = DataLoader(dataset_name= hp.dataset_name,
                            hg_dir=hp.huggingface_data_dir, 
                            hg_split_name=hp.huggingface_split_name,
                            disk_dir=hp.data_save_dir)

    model = NeoXLoraModelLoader(base_model_name = hp.base_model,
                                base_model_version= hp.base_model_version)


    dataloader.preprocess(model.tokenizer)


    model.train(dataloader.train_eval_dataset, hp)
    model.save(hp.lora_save_path)
    wandb.finish()




if __name__ == "__main__":
    fire.Fire(finetune)