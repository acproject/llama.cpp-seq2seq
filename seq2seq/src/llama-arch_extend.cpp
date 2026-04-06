#include "llama-arch_extend.h"

#include <map>

static const std::map<llm_arch_extend, const char *> LLM_ARCH_EXTEND_NAMES = {
    { LLM_ARCH_NLLB, "nllb" },
};

const char * llm_arch_extend_name(llm_arch arch) {
    if (!llm_arch_is_extend(arch)) {
        return nullptr;
    }

    auto it = LLM_ARCH_EXTEND_NAMES.find(static_cast<llm_arch_extend>(static_cast<int>(arch)));
    if (it == LLM_ARCH_EXTEND_NAMES.end()) {
        return nullptr;
    }

    return it->second;
}

std::set<llm_tensor> llm_get_tensor_names_extend(llm_arch arch) {
    if (!llm_arch_is_extend(arch)) {
        return {};
    }

    switch (static_cast<llm_arch_extend>(static_cast<int>(arch))) {
        case LLM_ARCH_NLLB:
            return {
                LLM_TENSOR_TOKEN_EMBD,
                LLM_TENSOR_POS_EMBD,
                LLM_TENSOR_OUTPUT,
                LLM_TENSOR_DEC_OUTPUT_NORM,
                LLM_TENSOR_DEC_ATTN_NORM,
                LLM_TENSOR_DEC_ATTN_Q,
                LLM_TENSOR_DEC_ATTN_K,
                LLM_TENSOR_DEC_ATTN_V,
                LLM_TENSOR_DEC_ATTN_OUT,
                LLM_TENSOR_DEC_CROSS_ATTN_NORM,
                LLM_TENSOR_DEC_CROSS_ATTN_Q,
                LLM_TENSOR_DEC_CROSS_ATTN_K,
                LLM_TENSOR_DEC_CROSS_ATTN_V,
                LLM_TENSOR_DEC_CROSS_ATTN_OUT,
                LLM_TENSOR_DEC_FFN_NORM,
                LLM_TENSOR_DEC_FFN_DOWN,
                LLM_TENSOR_DEC_FFN_UP,
                LLM_TENSOR_ENC_OUTPUT_NORM,
                LLM_TENSOR_ENC_ATTN_NORM,
                LLM_TENSOR_ENC_ATTN_Q,
                LLM_TENSOR_ENC_ATTN_K,
                LLM_TENSOR_ENC_ATTN_V,
                LLM_TENSOR_ENC_ATTN_OUT,
                LLM_TENSOR_ENC_FFN_NORM,
                LLM_TENSOR_ENC_FFN_DOWN,
                LLM_TENSOR_ENC_FFN_UP,
            };
    }

    return {};
}
