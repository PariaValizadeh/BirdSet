from typing import Literal, Any
from collections import Counter
from src.datamodule.components.transforms import BaseTransforms
from src.datamodule.components.event_decoding import EventDecoding
from src.datamodule.components.transforms import BaseTransforms, EmbeddingTransforms
from src.datamodule.components.event_mapping import XCEventMapping
from src.datamodule.gadme_datamodule import GADMEDataModule
from .base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig
from datasets import load_dataset, load_from_disk, Audio, DatasetDict, Dataset, IterableDataset, IterableDatasetDict, Sequence
import logging
import torch
import os

from os.path import join


class filter():
    def __init__(self, class_numbers, k, label_column, mode:Literal["train", "valid"]) -> None:
        self.mode = mode
        self.k = k
        self.label_column = label_column
        self.class_dict = dict()
        self.handled_idx = []
        for name in range(class_numbers):
            self.class_dict[name] = 0
    
    def set_mode(self, mode:Literal["train", "valid"]):
        self.mode = mode
    
    def revert_one_hot(self, entry:list):
        for i in range(len(entry)):
            e = entry[i]
            if e == 1:
                return i
        return 0
    
    def call_train(self, x, idx):
        entry = x[self.label_column]
        if isinstance(entry, list):
            entry = self.revert_one_hot(entry)
        assert entry in self.class_dict.keys()
        if entry in self.class_dict.keys():
            if self.class_dict[entry] < self.k:
                self.class_dict[entry] = self.class_dict[entry]+1
                self.handled_idx.append(idx)
                return True
        return False
    
    def __call__(self, x, idx,**kwds: Any) -> Any:
        if self.mode == "train":
            return self.call_train(x, idx)
        elif self.mode == "valid":
            if idx in self.handled_idx:
                return False
            return True
        
        raise ValueError("Mode not supported")

# this class does not need a transforms!!!
class OfflineGADMEDataModule(GADMEDataModule):
    def __init__(
        self, 
        dataset: DatasetConfig = DatasetConfig(), 
        loaders: LoadersConfig = LoadersConfig(), 
        transforms: BaseTransforms = EmbeddingTransforms(),
        mapper: XCEventMapping = XCEventMapping(),
        embedding_model_name: Literal["Embedding_Yamnet", "Embedding_Perch_v4", "Embedding_VGGish", "Embedding_BirdNet_v2_4"] = "Embedding_BirdNet_v2_4",
        split_mode: int | None = None):
        super().__init__(dataset, loaders, transforms, mapper)
        self.embedding_model_name = embedding_model_name
        self.split_mode = split_mode
        logging.info(f"Using offline dataset for model {embedding_model_name}")
    
    def prepare_data(self):
        logging.info("Check if preparing has already been done.")
        if self._prepare_done:
            logging.info("Skip preparing.")
            return

        logging.info("Prepare Data")

        dataset = self._load_data()
        dataset = self._preprocess_data(dataset)
        dataset = self._create_splits(dataset)

        # set the length of the training set to be accessed by the model
        self.len_trainset = len(dataset["train"])        
        self._save_dataset_to_disk(dataset)
        
        # set to done so that lightning does not call it again
        self._prepare_done = True
    
    def _load_data(self, decode: bool = False):
        laod_path = join(self.dataset_config.data_dir, "embeddings", self.dataset_config.dataset_name, self.embedding_model_name)
        logging.info(f"Attempting to load dataset from disk at {laod_path}")
        dataset = load_from_disk(dataset_path=laod_path)
        
        # some stuff from baseclass
        if isinstance(dataset, IterableDataset |IterableDatasetDict):
            logging.error("Iterable datasets not supported yet.")
            return
        assert isinstance(dataset, DatasetDict | Dataset)
        dataset = self._ensure_train_test_splits(dataset)


        if self.dataset_config.subset:
            dataset = self._fast_dev_subset(dataset, self.dataset_config.subset)

        # dataset = dataset.cast_column(
        #     column="audio",
        #     feature=Audio(
        #         sampling_rate=self.dataset_config.sampling_rate,
        #         mono=True,
        #         decode=decode,
        #     ),
        # )
        return dataset        
    
    def _get_dataset(self, split):
        dataset_path = os.path.join(
            self.dataset_config.data_dir,
            f"{self.dataset_config.dataset_name}_processed", 
            split
        )

        dataset = load_from_disk(dataset_path)

        self.transforms.set_mode(split)

        if split == "train": # we need this for sampler, cannot be done later because set_transform
            self.train_label_list = dataset["labels"]

        # add run-time transforms to dataset
        dataset.set_transform(self.transforms, output_all_columns=False) 
        
        return dataset
    
    def _preprocess_data(self, dataset):
        match self.dataset_config.task:
            case "multiclass":
                dataset = self._preprocess_multiclass(dataset)
            case "multilabel":
                dataset = self._preprocess_multilabel(dataset)
            case _:
                raise ValueError(f"There was an issue trying to preprocess the data with task {self.dataset_config.task}")
        
        dataset["train"] = dataset["train"].select_columns(
            ["embeddings", "labels", "detected_events", "start_time", "end_time", "no_call_events"]
        )
        # maybe has to be added to test data to avoid two selections
        dataset["test"]= dataset["test"].select_columns(
            ["embeddings", "labels", "detected_events", "start_time", "end_time"]
        )
        return dataset
    
    def _preprocess_multiclass(self, dataset):
        dataset = DatasetDict({split: dataset[split] for split in ["train", "test"]})

        if self.dataset_config.class_weights_loss or self.dataset_config.class_weights_sampler:
            self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))

        dataset = dataset.rename_column("ebird_code", "labels")
        return dataset
    
    def _preprocess_multilabel(self, dataset):
        dataset = DatasetDict({split: dataset[split] for split in ["train", "test_5s"]})

        dataset = dataset.map(
            self._classes_one_hot,
            batched=True,
            batch_size=300,
            load_from_cache_file=False,
            num_proc=self.dataset_config.n_workers,
            fn_kwargs={"column_name": "ebird_code_multilabel"}
        )

        if self.dataset_config.class_weights_loss or self.dataset_config.class_weights_sampler:
            self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))

        dataset["test"] = dataset["test_5s"]
        dataset = dataset.rename_column("ebird_code_multilabel", "labels")
        return dataset

    def _create_splits(self, dataset: DatasetDict | Dataset):
        dataset.cleanup_cache_files()
        if self.split_mode is None or self.split_mode == 0:
            return super()._create_splits(dataset)
        else:
            return self._create_k_split(dataset)
    
    def _create_k_split(self, dataset: DatasetDict | Dataset):
        dataset.cleanup_cache_files()
        k = self.split_mode
        if isinstance(dataset, Dataset):
            test_size = 0.2*self.dataset_config.val_split
            train_test = dataset.train_test_split(test_size=test_size, shuffle=True, seed=self.dataset_config.seed)
            train, valid = self.create_split(k, train_test["train"], False)
            return DatasetDict({"train": train, "valid": valid, "test": train_test["test"]})
        elif isinstance(dataset, DatasetDict):
            # this is probably the default
            if "train" in dataset.keys() and "valid" in dataset.keys() and "test" in dataset.keys():
                # correct ds
                return dataset
            if "train" in dataset.keys() and "test" in dataset.keys():
                # we need to split here!
                train, valid = self.create_split(k, dataset["train"], True)
                return DatasetDict({"train": train, "valid": valid, "test": dataset["test"]})
            # if dataset has only one key, split it into train, valid, test
            elif "train" in dataset.keys() and "test" not in dataset.keys():
                return self._create_splits(dataset["train"])
            else: 
                return self._create_splits(dataset[list(dataset.keys())[0]])    
    
    def create_split(self, k, dataset: Dataset, shuffle:bool=False):
        if shuffle:
            dataset = dataset.shuffle(seed=self.dataset_config.seed)
        # print(dataset)
        c_numbers = self.dataset_config.n_classes
        f = filter(c_numbers, k, "labels", "train")
        train = dataset.filter(f, with_indices=True)
        # print(train)
        f.set_mode("valid")
        valid = dataset.filter(f, with_indices=True)
        # print(valid)
        if len(dataset) != (len(train) + len(valid)):
            raise ValueError(f"dataset with length {len(dataset)} does not equal sum of train {len(train)} and valid {len(valid)}")
        return train, valid