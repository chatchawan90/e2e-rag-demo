"""Memory-conscious inference for the same multilingual E5 model in the hosted demo."""
import threading
import os
from functools import lru_cache

import numpy as np

MODEL = 'intfloat/multilingual-e5-small'
REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
# The official int8 export keeps the original model's embedding space.
MODEL_FILE = 'onnx/model_qint8_avx512_vnni.onnx'
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _runtime():
    from huggingface_hub import hf_hub_download
    from sentencepiece import SentencePieceProcessor
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.add_session_config_entry('session.disable_prepacking', '1')
    # Fusion can materialize large full-precision weights and exceed free-host RAM.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    model = os.environ.get('ENVSEARCH_ONNX_MODEL') or hf_hub_download(MODEL, MODEL_FILE, revision=REVISION, token=False)
    tokenizer = SentencePieceProcessor(model_file=hf_hub_download(MODEL, 'sentencepiece.bpe.model', revision=REVISION, token=False))
    session = ort.InferenceSession(model, sess_options=options, providers=['CPUExecutionProvider'])
    return tokenizer, session


def encode(texts):
    result = []
    # One shared encoder for Streamlit and MCP, including initialization.
    with _lock:
        tokenizer, session = _runtime()
        inputs = {value.name for value in session.get_inputs()}
        max_tokens = int(os.environ.get('ENVSEARCH_ONNX_MAX_TOKENS', '512'))
        for text in texts:
            # XLM-R's fairseq IDs offset SentencePiece IDs by one; unk maps to 3.
            pieces = [3 if token == 0 else token + 1 for token in tokenizer.encode(text)]
            ids = np.asarray([[0, *pieces[:max_tokens - 2], 2]], dtype=np.int64)
            mask = np.ones_like(ids)
            feed = {'input_ids': ids, 'attention_mask': mask}
            if 'token_type_ids' in inputs:
                feed['token_type_ids'] = np.zeros_like(ids)
            hidden = session.run(None, feed)[0]
            weights = mask[..., None]
            pooled = (hidden * weights).sum(axis=1) / weights.sum(axis=1)
            pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
            result.append(pooled[0])
    return np.asarray(result, dtype=np.float32)
