"""Externalize ONNX weights during image build to avoid large protobuf startup copies."""
from pathlib import Path
import os

from huggingface_hub import hf_hub_download
import onnx

from envsearch.onnx_embeddings import MODEL, MODEL_FILE, REVISION

source = hf_hub_download(MODEL, MODEL_FILE, revision=REVISION, token=False)
hf_hub_download(MODEL, 'sentencepiece.bpe.model', revision=REVISION, token=False)
destination = Path(os.environ['ENVSEARCH_ONNX_MODEL'])
destination.parent.mkdir(parents=True, exist_ok=True)
model = onnx.load(source)
onnx.save_model(model, str(destination), save_as_external_data=True, all_tensors_to_one_file=True,
                location='weights.bin', size_threshold=1024)
