# %%
from src.datamodule.gadme_datamodule import GADMEDataModule
from src.datamodule.base_datamodule import DatasetConfig, LoadersConfig, LoaderConfig
from src.datamodule.components.transforms import TransformsWrapper, PreprocessingConfig, BatchTransformer
from src.datamodule.components.event_mapping import XCEventMapping
from src.datamodule.components.event_decoding import EventDecoding
from src.datamodule.components.feature_extraction import DefaultFeatureExtractor
from omegaconf import DictConfig

import src.modules.models.embedding_models.perch_tf_embedding_model as embed
import src.modules.models.embedding_models.embedding_classifier_model as classifier

# %%
birdnet = embed.BirdNetTfEmbeddingModel()

# %%
num_classes = 22
sample_rate = 48000
window_size_s = 3.0
learning_rate = 1e-3
num_epochs = 5

# %%
dataset_name = "DBD-research-group/gadme_v1"
cache_dir = "/Volumes/BigChongusF/Datasets/Huggingface/gadme_v1/data"
dataset_config = DatasetConfig(cache_dir, "high_sierras", dataset_name, "high_sierras", 42, num_classes, 3, 0.2, "multiclass", None, sample_rate)
loaders_config = LoadersConfig()
loaders_config.train = LoaderConfig(12, True, 6, True, False, True, 2)
loaders_config.valid = LoaderConfig(12, False)
loaders_config.test = LoaderConfig(12, False)
mapper = XCEventMapping(biggest_cluster=True,
                        event_limit=5,
                        no_call=True)
transforms_wrapper = BatchTransformer(
    task = "multiclass",
    sampling_rate=sample_rate,
    max_length=window_size_s)
dm = GADMEDataModule(dataset_config, loaders_config, transforms_wrapper, mapper)

# %%
dm.prepare_data()
dm.setup("fit")

# %%
from lightning import Trainer

# %%
model = classifier.TfEmbeddingClassifier(birdnet, num_classes, learning_rate, num_epochs)
model

# %%
loader = dm.train_dataloader()

# %%
trainer = Trainer()

# %%
birdnet.embedding_dimension

# %%
trainer.fit(model=model,
            datamodule=dm)


