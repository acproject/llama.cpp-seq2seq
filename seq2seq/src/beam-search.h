#pragma once

#include "llama.h"
#include <utility>
#include <vector>
#include <functional>

namespace llama_beam {
    // Configuation for beam search
    struct beam_search_params {
        int beam_size = 5;
        float length_penalty_alpha = 1.0f;
        int max_length = 200;
        bool early_stopping = true;
        int min_length = 1;
        float diversity_penalty = 0.0f;

        int top_k_per_beam = 0;
        float score_threshold = -1e9f;
        bool normalize_scores = true;
    };

    struct beam_hypothesis {
        std::vector<llama_token> tokens;
        float score;
        float normalize_score;
        llama_seq_id seq_id;
        bool finished;

        beam_hypothesis(): score(0.0f), normalize_score(0.0f), seq_id(-1), finished(false) {}
    };

    struct beam_candidate {
        beam_hypothesis hyp;
        int parent_beam_idx;
        llama_seq_id parent_seq_id;
        llama_token last_token;
        float token_log_prob;

        beam_candidate(): parent_seq_id(-1), parent_beam_idx(-1), last_token(-1), token_log_prob(0.0f) {}
    };

    struct beam_search_result {
        std::vector<beam_hypothesis> hypotheses;
        int n_steps;
        bool stopped_early;

        // Get best hypothesis
        const beam_hypothesis& best() const {
            return hypotheses.empty() ? *(beam_hypothesis*)nullptr : hypotheses[0];
        }
    };

class beam_search_engine {
public:
    beam_search_engine(
        llama_context* ctx,
        const beam_search_params& params
    );

    ~beam_search_engine();

    beam_search_result search(
        const std::vector<llama_token>& initial_tokens,
        std::function<bool(llama_token)> is_eos
    );
    void initialize(const std::vector<llama_token> & initial_tokens);
    bool step(std::function<bool(llama_token)> is_eos);  // Returns false when done
    beam_search_result get_result();
        

    using step_callback_t = std::function<void(int step, const std::vector<beam_hypothesis&>)>;
    void set_step_callback(step_callback_t callback);
private:
        llama_context* ctx_;
        beam_search_params params_;

        std::vector<beam_hypothesis> beam_;
        std::vector<beam_candidate> candidates_;

        int current_step_;
        bool initialized_;

        step_callback_t step_callback_;

        // Internal methods
        void expand_beams(std::function<bool(llama_token)> is_eos);
        void prune_candidates();
        void rearrange_kv_caches();
        float compute_sorce(const beam_hypothesis& hyp) const;
        float apply_length_penalty(float score, int length) const;

        // Helper 
        std::vector<std::pair<llama_token, float>> get_top_k_tokens(
            const float* logits,
            int n_vocab,
            int k
        ) const;
};

inline bool is_eos_token(llama_token token, const llama_vocab * vocab) {
    return llama_vocab_is_eog(vocab, token);
}

void print_hypothesis(
    const beam_hypothesis & hyp,
    const llama_vocab * vocab,
    const char * prefix = ""
);

// Compare hypotheses by score (for sorting)
inline bool compare_hypotheses_by_score(
    const beam_hypothesis & a,
    const beam_hypothesis & b
) {
    return a.normalized_score > b.normalized_score;
}

inline bool compare_candidates_by_score(
    const beam_candidate & a,
    const beam_candidate & b
) {
    return a.hyp.normalized_score > b.hyp.normalized_score;
}
}