import os
from  datasets  import  load_dataset, load_from_disk

class DataLoader:
    def __init__(self, dataset_name, hg_dir, hg_split_name, disk_dir, reload=False) -> None:
        if reload:
            os.rmdir(disk_dir)
        if os.path.exists(disk_dir):
            self.dataset = load_from_disk(disk_dir)
        else:
            data = load_dataset(dataset_name, data_dir=hg_dir, split=hg_split_name)
            data.save_to_disk(disk_dir)
            self.dataset = data
    

    def preprocess(self, tokenizer, seq_max_length):
        def tokenize_and_split(sample):
            return tokenizer(sample['content'], 
                             max_length=seq_max_length,
                             return_overflowing_tokens=True)

        tokenized_dataset = self.dataset.map(tokenize_and_split, batched = True, remove_columns=self.dataset.column_names)

        train_eval = tokenized_dataset.train_test_split(test_size=0.1)
        self.train_eval_dataset = train_eval

            

            
