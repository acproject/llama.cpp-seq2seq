#pragma once

#include "llama-arch.h"

#include <set>

enum llm_arch_extend : int {
    LLM_ARCH_EXTEND_BASE = static_cast<int>(LLM_ARCH_UNKNOWN) + 1,
    LLM_ARCH_NLLB        = LLM_ARCH_EXTEND_BASE,
};

inline llm_arch llm_arch_from_extend(llm_arch_extend arch) {
    return static_cast<llm_arch>(static_cast<int>(arch));
}

inline bool llm_arch_is_extend(llm_arch arch) {
    return static_cast<int>(arch) >= LLM_ARCH_EXTEND_BASE;
}

const char * llm_arch_extend_name(llm_arch arch);
std::set<llm_tensor> llm_get_tensor_names_extend(llm_arch arch);
