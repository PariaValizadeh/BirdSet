import numpy as np
import tensorflow
from src.bird_sep_paper import model_utils
from os.path import join as path_join

tf = tensorflow.compat.v1

classifier_path = "/Users/moritzrichert/Models/birdseperation/birbsep_paper"

model_path = "sierras"

model_path = path_join(classifier_path, model_path, "run_00")
classy = model_utils.load_classifier_state(model_path)

fake_audio = np.zeros([1, 5*22050])
# For the `lorikeet` model, there are 87 output classes, so the hints should
# have shape [Batch, 87]. The Sierras model has 89 output species.
hints = np.ones([1, 87])

# Now call the model and get embeddings.
embeddings = model_utils.model_embed(
    fake_audio, classy, hints, 'hidden_embedding')

print(embeddings)