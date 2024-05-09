import pathlib
from huggingface_hub import snapshot_download
# NOTE: this needs to be in namespace to load transcoder
from transcoders_slim import sae_training # noqa
from transcoders_slim.sae_training.sparse_autoencoder import SparseAutoencoder

REPO_ID = "pchlenski/gpt2-transcoders"
FILENAMES = [
    f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.ln2.hook_normalized_24576.pt"
        for layer in range(12)
]

def download() -> list[pathlib.Path]:
    folder_path = pathlib.Path(snapshot_download(repo_id=REPO_ID))
    file_paths = [folder_path / filename for filename in FILENAMES]
    return file_paths

def load_pretrained():
    file_paths = download()
    transcoder = SparseAutoencoder.load_from_pretrained(str(file_paths[0]))
    print(transcoder)

load_pretrained()