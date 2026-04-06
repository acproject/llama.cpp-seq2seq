#ifndef GGML_EXTEND_H
#define GGML_EXTEND_H
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>
#include <unordered_map>

#include "ggml-alloc.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml_extend.hpp"

#ifdef SD_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef SD_USE_METAL
#include "ggml-metal.h"
#endif

#include "rng.hpp"
#include "util.h"

#define EPS 1e-05f

#ifndef __STATIC_INLINE__
#define __STATIC_INLINE__ static inline
#endif

#ifndef SD_UNUSED
#define SD_UNUSED(x) (void)(x)
#endif

__STATIC_INLINE__ int align_up_offset(int n, int multiple) {
    return (multiple - n % multiple) % multiple;
}

__STATIC_INLINE__ int align_up(int n, int multiple) {
    return n + align_up_offset(n, multiple);
}

__STATIC_INLINE__ void ggml_log_callback_default(ggml_log_level level, const char *text, void *) {
    switch (level) {
        case GGML_LOG_LEVEL_ERROR:
            LOG_DEBUG(text);
            break;
        case GGML_LOG_LEVEL_WARN:
            LOG_WARN(text);
            break;
        case GGML_LOG_LEVEL_INFO:
            LOG_INFO(text);
            break;
        case GGML_LOG_LEVEL_DEBUG:
            LOG_DEBUG(text);
            break;
        default:
            LOG_DEBUG(text);
    }
}

static_assert(GGML_MAX_NAME >= 128 , "GGML_MAX_NAME must be at least 128");

// n-mode tensor-matrix product
// example: 2-mode product
// A: [ne03, k , ne01, ne00]
// B: k rows, m columns => [k, m]
// result is [ne03, m, ne01, ne00]

#endif
