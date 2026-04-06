#pragma once

#include "ggml.h"
#include "gguf.h"


/// ```
/// 文本输入 -> CLIP Text Encoder -> Text Embedding (77 * 1024)
///                                             ↓
/// 随机噪声 -》 U-Net （扩散去噪， 20-50步） -> 去噪声 Latent
///                                             ↓
///                        VAE Decoder -> 图像输出 (512 * 512 或 768 * 768)
/// ```

// CLIP ViT-H/14 结构
struct CLIPTextEncoder {
    // Token Embedding: vocab_size * hidden_dim
    struct ggml_tensor* token_embedding; // [49408, 1024]

    // Postion Embedding: max_length * hidden_dim
    struct ggml_tensor* position_embedding; // [77, 1024]

    // Transformer Layers (32 layers for ViT-H)
    struct CLIPLayer {
        // Self-Attention
        struct ggml_tensor* attention_qkv_weight; // [1024, 3*1024] 
        struct ggml_tensor* attention_qkv_bias;
        struct ggml_tensor* attention_out_weight; // [1024, 1024]
        struct ggml_tensor* attention_out_bias;

        // Layer Norm 1 & 2
        struct ggml_tensor* ln1_weight;
        struct ggml_tensor* ln1_bias;
        struct ggml_tensor* ln2_weight;
        struct ggml_tensor* ln2_bias;

        // FFN (MLP)
        struct ggml_tensor* ffn_up_weight; // [1024, 4096]
        struct ggml_tensor* ffn_up_bias;
        struct ggml_tensor* ffn_down_weight; // [4096, 1024]
        struct ggml_tensor* ffn_down_bias;
    } layers[32];

    // Final Layer Norm
    struct ggml_tensor* final_ln_weight;
    struct ggml_tensor* final_ln_bias;

    // Text Projection (for pooled output)
    struct ggml_tensor* text_projection; // [1024, 1024]
};

// SD UNet 模型结构
struct UNetModel {
    // input projection 
    // Conv IN ; latent from 4ch to 320ch
    struct ggml_tensor* conv_in_weight;  // [320, 4, 3, 3]
    struct ggml_tensor* conv_in_bias;  // [320]

    // Time embedding
    struct ggml_tensor* time_embedding_linear1_weight;  // [320, 1280]
    struct ggml_tensor* time_embedding_linear1_bias;  // [1280]
    struct ggml_tensor* time_embedding_linear2_weight;  // [1280, 1280]
    struct ggml_tensor* time_embedding_linear2_bias;  // [1280]

    // Text Conditioning Projection
    struct ggml_tensor* proj_in_weight;  // [1024, 1024] or cross_attn dims
    
    // down sampling encoder
    // downblocks 1 : 320ch,  2 ResBlocks + 2 CrossAttnBlocks
    // downblocks 2 : 640ch,  2 ResBlocks + 2 CrossAttnBlocks
    // downblocks 3 : 1280ch, 2 ResBlocks + 2 CrossAttnBlocks
    // downblocks 4 : 1280ch, 2 ResBlocks (no CrossAttnBlocks)

    // middle block
    // middleblock  1280ch, 1 ResBlocks + 1 CrossAttnBlocks + 1 ResBlocks

    // up sampling decoder
    // upblocks 1 : 1280ch, 3 ResBlocks + 3 CrossAttnBlocks
    // upblocks 2 : 1280ch,  3 ResBlocks + 3 CrossAttnBlocks
    // upblocks 3 : 640ch,  3 ResBlocks + 3 CrossAttnBlocks
    // upblocks 4 : 320ch,  3 ResBlocks (no CrossAttnBlocks)

    // output projection
    struct ggml_tensor* conv_norm_out_weight;  // [320]
    struct ggml_tensor* conv_norm_out_bias;  // [320]
    struct ggml_tensor* conv_out_weight;  // [4, 320, 3, 3]
    struct ggml_tensor* conv_out_bias;  // [4]

};


// ResBlock 结构
struct ResBlock {
    // Group Normalization
    struct ggml_tensor* norm1_weight; // [in_channels]
    struct ggml_tensor* norm1_bias; 

    // Conv 3x3
    struct ggml_tensor*  conv1_weight; // [out_channels, in_channels, 3, 3]
    struct ggml_tensor*  conv1_bias;

    // Time Embedding Projection
    struct ggml_tensor* time_emb_proj_weight; // [out_channels, temb_channels]
    struct ggml_tensor* time_emb_proj_bias;
    // Group Normalization 2
    struct ggml_tensor* norm2_weight; // [in_channels]
    struct ggml_tensor* norm2_bias; 
    // Conv 3x3
    struct ggml_tensor*  conv2_weight; // [out_channels, in_channels, 3, 3]
    struct ggml_tensor*  conv2_bias;

    // Skip Connection Conv (if in_channels != out_channels)
    struct ggml_tensor* conv_shortcut_weight; // [out_channels, in_channels, 3, 3]
    struct ggml_tensor* conv_shortcut_bias;
};

// CrossAttentionBlock 结构
struct CrossAttention {
    // Group Normalization
    struct ggml_tensor* norm_weight; // [in_channels]
    struct ggml_tensor* norm_bias; 

  // Q, K ,V projections
  struct ggml_tensor * to_q_weight; // [inner_dim, query_dim]
  struct ggml_tensor * to_k_weight; // [inner_dim, context_dim]
  struct ggml_tensor * to_v_weight; // [inner_dim, context_dim]

  // out
  struct ggml_tensor * to_out_weight; // [query_dim, inner_dim]
  struct ggml_tensor * to_out_bias;
};

struct VAEDecoder {
    // Conv IN
    struct ggml_tensor* conv_in_weight;  // [512, 4, 3, 3]
    struct ggml_tensor* conv_in_bias;  // [512]

    // ResBlocks + upsample
    // Block 1: 512ch, 3 ResBlocks
    // Block 2: 512ch, -> 256ch, 3 ResBlocks + UpSample
    // Block 3: 256ch, -> 128ch, 3 ResBlocks + UpSample
    // Block 4: 128ch, -> 128ch, 3 ResBlocks + UpSample

    // Conv OUT
    struct ggml_tensor* conv_norm_out_weight;  // [128]
    struct ggml_tensor* conv_norm_out_bias;  // [128]
    struct ggml_tensor* conv_out_weight;  // [3, 128, 3, 3]
    struct ggml_tensor* conv_out_bias;  // [3]

};

// VAE ResBlock 结构
struct  VAEResBlock {
    // Group Normalization
    struct ggml_tensor* norm1_weight; // [in_channels]
    struct ggml_tensor* norm1_bias; 

    // Conv 3x3
    struct ggml_tensor*  conv1_weight; // [out_channels, in_channels, 3, 3]
    struct ggml_tensor*  conv1_bias;

    // Group Normalization 2
    struct ggml_tensor* norm2_weight; // [in_channels]
    struct ggml_tensor* norm2_bias; 
    // Conv 3x3
    struct ggml_tensor*  conv2_weight; // [out_channels, in_channels, 3, 3]
    struct ggml_tensor*  conv2_bias;

    // Skip Connection Conv (if in_channels != out_channels)
    struct ggml_tensor* conv_shortcut_weight; // [out_channels, in_channels, 3, 3]
    struct ggml_tensor* conv_shortcut_bias;
};
