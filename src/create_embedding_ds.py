import os
from typing import Any, Literal
from datasets import Dataset, DatasetDict, disable_caching
import rootutils
import hydra
import lightning as L 
from omegaconf import OmegaConf
from src import utils
import pyrootutils 
from src.datamodule.base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig
from src.datamodule.components.event_decoding import EventDecoding
from src.datamodule.components.feature_extraction import DefaultFeatureExtractor
from src.datamodule.components.transforms import BaseTransforms
from src.datamodule.components.event_mapping import XCEventMapping
from src.datamodule.gadme_datamodule import GADMEDataModule


log = utils.get_pylogger(__name__)
#rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base":None,
    #"config_path": "../configs",
    "config_path": str(root / "configs"),
    "config_name": "main.yaml"
}

class MyTransforms(BaseTransforms):
    def __init__(self, other: BaseTransforms = BaseTransforms()) -> None:
        super().__init__(other.task, other.sampling_rate, other.max_length, other.event_decoder, other.feature_extractor)
    
    def set_task(self, task):
        self.task = task
    
    def transform_labels(self, batch):
        return batch["labels"]

class DownloadableData(GADMEDataModule):
    def __init__(
        self,
        dataset: DatasetConfig = DatasetConfig(),
        loaders: LoadersConfig = LoadersConfig(),
        transforms: BaseTransforms = BaseTransforms(),
        mapper: XCEventMapping = XCEventMapping()
        ) -> None:
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=MyTransforms(transforms),
            mapper=mapper
        )
    
    def _preprocess_data(self, dataset):
        dataset["train"] = dataset["train"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )
        
        if self.dataset_config.class_weights_loss or self.dataset_config.class_weights_sampler:
                self.num_train_labels = self._count_labels((dataset["train"]["ebird_code"]))
        
        dataset = dataset.rename_column("ebird_code", "labels")
        # dataset = dataset.rename_column("ebird_code_multilabel", "labels")

        dataset["train"] = dataset["train"].select_columns(
            ["filepath", "labels", "detected_events", "start_time", "end_time", "no_call_events", "ebird_code_multilabel"]
        )
        # maybe has to be added to test data to avoid two selections
        dataset["test"]= dataset["test"].select_columns(
            ["filepath", "labels", "detected_events", "start_time", "end_time", "ebird_code_multilabel"]
        )
        dataset["test_5s"]= dataset["test_5s"].select_columns(
            ["filepath", "ebird_code_multilabel", "detected_events", "start_time", "end_time"]
        )
        return dataset
    
    def _create_splits(self, dataset: DatasetDict | Dataset):
        if isinstance(dataset, DatasetDict):
            return dataset
        raise ValueError(f"Type of dataset is {type(dataset)}")
    
    def setup(self, stage=None):
        self.train_dataset = self._get_dataset("train")
        self.test_dataset = self._get_dataset("test")
        self.valid_dataset = self._get_dataset("test_5s")
        self.valid_dataset = self.valid_dataset.rename_column("ebird_code_multilabel", "labels")
        self.test_5s_dataset = self.valid_dataset
    
    def set_task(self, task):
        self.transforms.set_task(task)
    

class EmbeddingCreator():
    def __init__(self, embedding_model) -> None:
        self.embedding_model = embedding_model
    
    def embed(self, x):
        x = x["input_values"]
        embeddings = self.embedding_model.forward_embed(x, device=None)
        return {"embeddings": embeddings}
    
    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        return self.embed(x)

@utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def main(cfg):
    log.info('Using config: \n%s', OmegaConf.to_yaml(cfg))

    log.info(f"Dataset path: <{os.path.abspath(cfg.paths.dataset_path)}>")
    os.makedirs(cfg.paths.dataset_path, exist_ok=True)
    
    embeddings_path = os.path.join(cfg.paths.dataset_path, "embeddings")
    log.info(f"Embeddings Path: <{os.path.abspath(embeddings_path)}>")
    os.makedirs(embeddings_path, exist_ok=True)

    log.info(f"Log path: <{os.path.abspath(cfg.paths.log_dir)}>")
    os.makedirs(cfg.paths.log_dir, exist_ok=True)

    log.info(f"Seed everything with <{cfg.seed}>")
    L.seed_everything(cfg.seed)
    #log.info(f"Instantiate logger {[loggers for loggers in cfg['logger']]}")
    
    # Setup data
    log.info(f"Instantiate datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data() # has to be called before model for len_traindataset!
    datamodule.setup()
    
    log.info(f"Instantiate Tf Module {cfg.module.network.model.embedding_model._target_}")
    embedding_model = hydra.utils.instantiate(cfg.module.network.model.embedding_model)
    
    train = datamodule.train_dataset
    test = datamodule.test_dataset
    test5s = datamodule.test_5s_dataset
    
    mapper = EmbeddingCreator(embedding_model)
    
    train = train.map(mapper, batched=True, batch_size=100)
    train = train.rename_column("labels", "ebird_code")
    test = test.map(mapper, batched=True, batch_size=100)
    test = test.rename_column("labels", "ebird_code")
    # datamodule.set_task("multiclass")
    test5s = test5s.map(mapper, batched=True, batch_size=100)
    test5s = test5s.rename_column("labels", "ebird_code_multilabel")
    
    dataset = DatasetDict({"train": train, "test": test, "test_5s": test5s})
    # dataset.rename_column("embeddings", "audio")
    dataset.reset_format()
    ds_path = os.path.join(embeddings_path, cfg.datamodule.dataset.dataset_name, cfg.module.network.model_name)
    log.info(f"Saving to: <{os.path.abspath(ds_path)}>")
    os.makedirs(ds_path, exist_ok=True)
    
    # print(dataset["train"][0])
    
    print(dataset)

    log.info("Saving now")
    dataset.save_to_disk(ds_path)

    print(dataset.cleanup_cache_files())
    
    utils.close_loggers()
    

if __name__ == "__main__":   
    disable_caching() 
    main()

    