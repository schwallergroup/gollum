from dataclasses import dataclass
import os
from typing import Optional
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()

import numpy as np

from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoTokenizer,
    AutoModel,
)
from functools import partial
import torch.nn.functional as F


from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

from openai import OpenAI

client = OpenAI()

from transformers import AutoTokenizer
from gollum.featurization.utils.pooling import average_pool, last_token_pool, weighted_average_pool





@lru_cache(maxsize=None)
def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return (
        client.embeddings.create(input=[text], model=model).data[0].embedding
    )


def ada_embeddings(texts, model="text-embedding-ada-002"):
    """
    Get ADA embeddings for a list of texts.

    :param texts: List of texts to be embedded
    :type texts: list of str
    :param model: Model name to use for embedding (default is "text-embedding-ada-002")
    :type model: str
    :return: NumPy array of ADA embeddings
    """
    get_embedding_with_model = partial(get_embedding, model=model)

    with ProcessPoolExecutor() as executor:
        embeddings = list(
            tqdm(
                executor.map(get_embedding_with_model, texts),
                total=len(texts),
                desc="Getting Embeddings",
            )
        )
    return np.array(embeddings)


def ada_embeddings_3(texts, model="text-embedding-3-small"):
    return ada_embeddings(texts, model=model)


from transformers import T5Tokenizer, T5EncoderModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5EncoderModel, T5Config
from transformers import LlamaModel, LlamaConfig



@dataclass
class ModelConfig:
    name: str
    config_class: Optional[any] = None
    model_class: Optional[any] = None
    dropout_field: str = "dropout_rate"


MODEL_CONFIGS = {
    "t5-base": ModelConfig("t5-base", T5Config, T5EncoderModel),
    "GT4SD/multitask-text-and-chemistry-t5-base-augm": ModelConfig(
        "GT4SD/multitask-text-and-chemistry-t5-base-augm",
        T5Config,
        T5EncoderModel,
    ),
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp": ModelConfig(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        LlamaConfig,
        LlamaModel,
        "attn_dropout",
    ),
}


def get_model_and_tokenizer(model_name: str, device: str='cuda'):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    if model_config := MODEL_CONFIGS.get(model_name):
        config = model_config.config_class.from_pretrained(model_name)
        setattr(config, model_config.dropout_field, 0)
        model = model_config.model_class.from_pretrained(
            model_name, config=config
        ).to(device)
    else:
        model = AutoModel.from_pretrained(
            model_name, device_map=device, trust_remote_code=True
        )

    return model, tokenizer


def get_tokens(
    texts,
    model_name="WhereIsAI/UAE-Large-V1",
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    print(model_name, "for get tokens")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_batches = []
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    input_ids_padded = pad_sequence(
        [torch.tensor(ids) for ids in encoded_input.input_ids],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_masks_padded = pad_sequence(
        [torch.tensor(mask) for mask in encoded_input.attention_mask],
        batch_first=True,
        padding_value=0,
    )
    all_encoded_inputs = torch.cat(
        [input_ids_padded, attention_masks_padded], dim=1
    )
    return all_encoded_inputs.cpu().numpy()


def get_huggingface_embeddings(
    texts,
    model_name="tiiuae/falcon-7b",
    max_length=512,
    batch_size=8,
    pooling_method="cls",
    prefix=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    normalize_embeddings=False,
):
    """
    General function to get embeddings from a HuggingFace transformer model.
    """
    print(f"featurizing with {model_name}")
    model, tokenizer = get_model_and_tokenizer(model_name, device)
    left_padding = tokenizer.padding_side == "left"
    model.eval()

    # optionally add prefix to each text
    if prefix:
        texts = [prefix + text for text in texts]

    pooling_functions = {
        "average": average_pool,
        "cls": lambda x, _: x[:, 0],
        "last_token_pool": partial(last_token_pool, left_padding=left_padding),
        "weighted_average": weighted_average_pool,
    }

    embeddings_list = []
    for i in tqdm(
        range(0, len(texts), batch_size), desc=f"Processing with {model_name}"
    ):
        batch_texts = texts[i : i + batch_size]
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded_input)
            pooled = pooling_functions[pooling_method](
                outputs.last_hidden_state, encoded_input["attention_mask"]
            )

            if normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=1)
            embeddings_list.append(pooled.cpu().numpy())

        torch.cuda.empty_cache()

    return np.concatenate(embeddings_list, axis=0)


def get_sentence_transformer_embeddings(
    texts, model_name="bigscience/sgpt-bloom-7b1-msmarco", batch_size=32
):
    model = SentenceTransformer(model_name)
    embeddings_list = []
    for i in tqdm(
        range(0, len(texts), batch_size), desc=f"Processing with {model_name}"
    ):
        batch_texts = texts[i : i + batch_size]
        embeddings = model.encode(batch_texts)
        embeddings_list.append(embeddings)

    return np.concatenate(embeddings_list, axis=0)


def instructor_embeddings(
    texts,
    model_name="hkunlp/instructor-xl",
    instruction="Represent the chemistry procedure: ",
    normalize=False,
):
    """
    Get Instructor embeddings for a list of texts.

    :param texts: List of texts to be embedded
    :type texts: list of str
    :param model_name: Pretrained model name to use for embedding
    :type model_name: str
    :param instruction: Instruction string for the embedding task
    :type instruction: str
    :return: NumPy array of Instructor embeddings
    """
    # Load the INSTRUCTOR model
    model = INSTRUCTOR(model_name).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 32
    sentence_embeddings_list = []
    paired_texts = [[instruction, text] for text in texts]

    for i in tqdm(range(0, len(paired_texts), batch_size)):
        batch_embeddings = model.encode(
            paired_texts[i : i + batch_size], normalize_embeddings=normalize
        )
        sentence_embeddings_list.append(batch_embeddings)


    return np.concatenate(sentence_embeddings_list, axis=0)
















