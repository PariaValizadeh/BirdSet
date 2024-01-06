from typing import Literal
from collections import Counter
from src.datamodule.components.transforms import BaseTransforms
from src.datamodule.components.event_decoding import EventDecoding
from src.datamodule.components.transforms import BaseTransforms, EmbeddingTransforms
from src.datamodule.components.event_mapping import XCEventMapping
from src.datamodule.gadme_datamodule import GADMEDataModule
from .base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig
from datasets import load_dataset, load_from_disk, Audio, DatasetDict, Dataset, IterableDataset, IterableDatasetDict
import logging
import torch
import os

from os.path import join

# this class does not need a transforms!!!
class OfflineGADMEDataModule(GADMEDataModule):
    def __init__(
        self, 
        dataset: DatasetConfig = DatasetConfig(), 
        loaders: LoadersConfig = LoadersConfig(), 
        transforms: BaseTransforms = EmbeddingTransforms(),
        mapper: XCEventMapping = XCEventMapping(),
        embedding_model_name: Literal["Embedding_Yamnet", "Embedding_Perch", "Embedding_VGGish", "Embedding_BirdNet_v2_4"] = "Embedding_BirdNet_v2_4"):
        super().__init__(dataset, loaders, transforms, mapper)
        self.embedding_model_name = embedding_model_name
    
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
        )

        if self.dataset_config.class_weights_loss or self.dataset_config.class_weights_sampler:
            self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))

        dataset["test"] = dataset["test_5s"]
        dataset = dataset.rename_column("ebird_code_multilabel", "labels")
        return dataset
        