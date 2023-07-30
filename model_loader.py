from transformers import GPTNeoXForCausalLM, AutoTokenizer, Trainer, TrainingArguments,DataCollatorForLanguageModeling 
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import os
from utils import tprint


class NeoXLoraModelLoader:
    def __init__(self, base_model_name, base_model_version, local_files_only = True) -> None:
        model_pure_name = os.path.split(base_model_name)[1]
        model_cache_dir = os.path.join("model", model_pure_name, base_model_version)

        local_files_only = os.path.exists(model_cache_dir) and local_files_only
        model = GPTNeoXForCausalLM.from_pretrained(
          base_model_name, 
          revision=base_model_version, 
          cache_dir= model_cache_dir,
          local_files_only=local_files_only
        )

        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query_key_value"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(model, config)


        self.tokenizer = AutoTokenizer.from_pretrained(
          base_model_name,
          revision=base_model_version,
          cache_dir= model_cache_dir,
          local_files_only=local_files_only,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def train(self, train_eval_dataset, hp):
        self.trainer = Trainer(
            model=self.model,
            train_dataset=train_eval_dataset['train'],
            eval_dataset=train_eval_dataset['test'],
            #compute_metrics=compute_metrics,
            #preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            args=TrainingArguments(
                per_device_train_batch_size= hp.per_device_train_batch_size,
                gradient_accumulation_steps= hp.gradient_accumulation_steps,
                eval_steps= hp.eval_steps,
                report_to="wandb",
                evaluation_strategy='steps',
                warmup_steps= hp.warmup_steps,
                max_steps= hp.max_steps,
                learning_rate= hp.learning_rate,
                logging_steps= hp.logging_steps,
                output_dir= hp.lora_save_path,
                optim= hp.opt_method, 
            ),
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        self.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        result = self.trainer.train()
        return result
    
    def save(self, lora_save_path):
        model_to_save = self.trainer.model.module if hasattr(self.trainer.model, 'module') else self.trainer.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(lora_save_path)

