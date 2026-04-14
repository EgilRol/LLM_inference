// You can set your configuration here...
// Read the manual for our assumption for directory of model weights and
// tokenizer files.
#pragma once

#include <string>

const std::string TOKENIZER_PATH = "assets/llama3/token.model";
const std::string WEIGHTS_DIR_PATH =
    "/shared/home/egr776/project/LLM_inference/data";
const std::string EMBED_WEIGHTS_PATH = WEIGHTS_DIR_PATH + "/embed_tokens.bin";

const int EMBEDDING_DIM = 4096;
const float RMS_NORM_EPSILON = 1e-5f;
