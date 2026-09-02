#pragma once

#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// SSD streaming of MoE routed expert weights
//
// Streamed layers do not materialize their ffn_*_exps tensors; instead each weight gets a
// device-side cache tensor of n_slots expert slabs, filled on demand from the GGUF file by an
// id-remapping custom op that runs on the CPU right after the router top-k. The remap only
// changes which cache slot an expert id resolves to - it never changes which experts the router
// selected, so streaming affects latency, not outputs.
//
// Missing experts are loaded by a pool of I/O threads while the remap op waits; eviction is by
// decaying route hotness with an LRU tiebreak. Reads are buffered by default, or O_DIRECT with
// LLAMA_MOE_STREAM_DIRECT=1 (bypasses the page cache; recommended when the model far exceeds RAM).
//
// note: multiple contexts decoding the same streamed model concurrently are not supported -
// one context can evict slots referenced by the other's in-flight graph.

struct llama_moe_stream;

enum llama_moe_stream_slot_state : uint8_t {
    LLAMA_MOE_STREAM_SLOT_EMPTY    = 0,
    LLAMA_MOE_STREAM_SLOT_LOADING  = 1, // reserved, load queued or in flight
    LLAMA_MOE_STREAM_SLOT_RESIDENT = 2,
};

// one streamed weight tensor (gate/up/down or fused gate_up) of one layer
struct llama_moe_stream_weight {
    ggml_tensor * cache = nullptr; // cache tensor {ne0, ne1, n_slots}

    uint16_t file_idx  = 0; // GGUF split file index
    size_t   offs      = 0; // file offset of the full exps tensor data
    size_t   nb_expert = 0; // bytes per expert slab
};

struct llama_moe_stream_layer;

struct llama_moe_stream_load {
    int32_t expert = -1;
    int32_t victim = -1;
    int32_t slot   = -1;
};

struct llama_moe_stream_plan {
    std::vector<llama_moe_stream_load> loads;
    std::vector<int32_t> next_map;
    std::vector<int32_t> mapped_topk;
    std::vector<int32_t> required_slots;
};

// userdata of one wave's custom ops (multi-pass prefill): identifies which pass this is
struct llama_moe_stream_wave {
    llama_moe_stream_layer * sl   = nullptr;
    int32_t                  wave = -1;
};

// per-layer streaming state - also the userdata of the id-remapping custom op
struct llama_moe_stream_layer {
    llama_moe_stream * mgr = nullptr;

    int32_t  il       = -1;
    uint32_t n_expert = 0;
    uint32_t n_slots  = 0;

    std::vector<llama_moe_stream_weight> weights; // 2 (fused gate_up + down) or 3 entries

    // residency state, guarded by mgr->mtx
    std::vector<int32_t>                 slot_expert;   // [n_slots] expert id or -1
    std::vector<uint8_t>                 slot_state;    // [n_slots] llama_moe_stream_slot_state
    std::vector<uint8_t>                 slot_claimed;  // [n_slots] a worker owns the load
    std::vector<uint64_t>                slot_gen;      // [n_slots] reservation generation
    std::vector<int64_t>                 slot_last_use; // [n_slots] LRU stamps
    std::unordered_map<int32_t, int32_t> expert_slot;   // RESIDENT and LOADING entries
    std::vector<int32_t>                 expert_map;    // [n_expert] committed RESIDENT mapping

    std::vector<uint32_t> route_hotness; // [n_expert] decayed selection counts, for eviction
    std::vector<uint8_t>  seen;          // [n_expert] for cold-miss attribution
    int64_t use_counter = 0;

    ggml_backend_t shadow_backend = nullptr;
    const ggml_backend_moe_stream_cache_ops * shadow_ops = nullptr;
    std::vector<int32_t> shadow_available_map;

    ggml_backend_t decode_backend = nullptr;
    const ggml_backend_moe_stream_cache_ops * decode_ops = nullptr;
    std::vector<int32_t> decode_available_map;

    // scratch for the remap callback
    std::vector<int32_t> uniq;
    std::vector<uint8_t> touched;
    std::vector<uint8_t> keep;         // [n_slots] slots the current call must not evict
    std::vector<int32_t> demand_slots; // slots the current call waits on
    llama_moe_stream_plan plan;

    // wave plan for multi-pass prefill (guarded by mgr->mtx): the touched experts are split into
    // plan_n_waves passes of at most plan_capacity experts each, run one pass at a time
    uint32_t plan_capacity  = 0;  // experts per wave, set at graph build
    uint32_t plan_n_waves   = 0;  // waves of the current call
    int32_t  plan_next_wave = -1; // wave expected to run next (ordering guard)
    std::vector<uint8_t> expert_wave; // [n_expert] wave each touched expert belongs to, 0xff = untouched
    std::vector<int32_t> plan_pool;   // resident slots the masked-out pairs of this wave park on
    std::vector<int32_t> pool_used;   // scratch: pool slots already used in the current token row

    std::vector<std::unique_ptr<llama_moe_stream_wave>> wave_ud; // stable per-wave op userdata

    // stable userdata for wave w (grows lazily); called at graph build time only
    llama_moe_stream_wave * wave_userdata(int32_t wave, uint32_t capacity);

    // whether the exps tensors passed to build_moe_ffn are this layer's cache tensors
    // (e.g. grovemoe evaluates a second, unstreamed expert group on the same layer index)
    bool matches(const ggml_tensor * gate, const ggml_tensor * up,
                 const ggml_tensor * down, const ggml_tensor * gate_up) const;
};

// one queued expert load
struct llama_moe_stream_work {
    llama_moe_stream_layer * sl = nullptr;

    int32_t  expert = -1;
    int32_t  slot   = -1;
    uint64_t gen    = 0; // stale unless it matches slot_gen[slot]
    int64_t  queued_us = 0;
};

struct llama_moe_stream_token_stats {
    int64_t n_hit               = 0;
    int64_t n_miss              = 0;
    int64_t n_shadow_plans      = 0;
    int64_t n_shadow_mismatches = 0;
    int64_t n_gpu_hit_plans      = 0;
    int64_t t_gpu_hit_segment_ns = 0;
    int64_t t_gpu_hit_planner_ns = 0;
    int64_t t_gpu_hit_wall_us    = 0;
    int64_t t_gpu_hit_sync_us    = 0;
    int64_t t_gpu_hit_cb_us      = 0;
    int64_t t_gpu_hit_prepare_us = 0;
    int64_t t_gpu_hit_commit_us  = 0;
    int64_t n_gpu_slow_plans      = 0;
    int64_t n_gpu_slow_loads      = 0;
    int64_t t_gpu_slow_segment_ns = 0;
    int64_t t_gpu_slow_planner_ns = 0;
    int64_t t_gpu_slow_wall_us    = 0;
    int64_t t_gpu_slow_sync_us    = 0;
    int64_t t_gpu_slow_cb_us      = 0;
    int64_t t_gpu_slow_prepare_us = 0;
    int64_t t_gpu_slow_load_us    = 0;
    int64_t t_gpu_slow_commit_us  = 0;
    int64_t n_gpu_slow_waiting     = 0;
    int64_t t_gpu_slow_resident_wait_us = 0;
    int64_t n_gpu_commit_carry     = 0;
    int64_t t_gpu_commit_carry_ns  = 0;
    int64_t n_worker_loads         = 0;
    int64_t t_worker_queue_us      = 0;
    int64_t t_worker_read_us       = 0;
    int64_t t_worker_upload_us     = 0;
};

struct llama_moe_stream {
    uint32_t n_slots      = 0; // expert cache slots per streamed layer
    int32_t  n_io_threads = 0;

    std::vector<std::unique_ptr<llama_moe_stream_layer>> layers; // [n_layer], null = not streamed

    llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct);
    ~llama_moe_stream();

    llama_moe_stream_layer * layer(int32_t il) const {
        return il >= 0 && (size_t) il < layers.size() ? layers[il].get() : nullptr;
    }

    // registers a streamed weight of layer il and returns its cache tensor
    ggml_tensor * create_cache_tensor(
            int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
            uint16_t file_idx, size_t offs);

    // allocate the cache tensor buffers (after all create_cache_tensor calls)
    void alloc_bufs(bool no_alloc);

    void bind_decode_backends(const std::vector<ggml_backend_t> & backends);
    void unbind_decode_backends();
    bool prepare_decode();
    void record_continuous_hits(size_t n_plans);
    bool eval_callback(
            ggml_backend_sched_t sched,
            ggml_tensor * tensor,
            bool ask,
            bool & requested);

    // reopen the GGUF files for streaming reads
    void open_files(const std::vector<std::string> & paths);

    size_t size_bufs() const;

    void print_stats() const;
    void token_stats_begin();
    llama_moe_stream_token_stats token_stats_end();

    bool use_direct_io = false; // O_DIRECT streaming reads (LLAMA_MOE_STREAM_DIRECT), no page cache

    llama_files files; // privately reopened GGUF files, same indices as the loader's

    size_t  max_nb_expert      = 0;
    int64_t hot_decay_interval = 0; // remap calls between route-hotness halvings (0 = no decay)

    std::vector<std::pair<ggml_backend_buffer_type_t, ggml_context_ptr>> ctxs; // one per buft
    std::vector<ggml_backend_buffer_ptr> bufs;
    std::vector<ggml_backend_ptr> shadow_backends;

    // load pool (queue and all layer residency state guarded by mtx)
    mutable std::mutex      mtx;
    std::condition_variable cv_work; // queued work or shutdown
    std::condition_variable cv_done; // a load committed or failed

    std::deque<llama_moe_stream_work> q_demand;

    std::vector<std::thread> workers;
    bool workers_started = false;
    bool shutting_down   = false;
    bool load_failed     = false;

    bool debug = false;
    bool shadow = false;
    bool gpu_decode_requested = false;
    bool gpu_decode = false;
    bool gpu_decode_continuous_requested = false;
    bool gpu_decode_continuous = false;
    bool gpu_decode_state_ready = false;
    bool gpu_decode_cpu_policy_stale = false;

    struct {
        int64_t n_calls     = 0; // remap invocations
        int64_t n_hit       = 0; // touched experts already resident or loading
        int64_t n_miss      = 0; // demand loads issued
        int64_t n_miss_cold = 0; // first-ever touch of an expert
        int64_t t_stall_us  = 0; // wait time in miss handling

        int64_t n_wave_calls     = 0; // wave-ids invocations (>= n_calls under multi-pass prefill)
        int64_t n_waves_run      = 0; // non-empty waves
        int64_t n_preload_issued = 0; // next-wave loads started during a wave's compute
        int64_t n_preload_ready  = 0; // wave experts already resident from the previous preload
        int64_t t_stall_wave_us  = 0; // wait time in wave miss handling

        int64_t n_shadow_plans      = 0;
        int64_t n_shadow_mismatches = 0;
    } stats;

    bool token_stats_active = false;
    llama_moe_stream_token_stats token_stats;

    // internals
    void start_workers_locked();
    void worker_loop();
    bool sync_decode_policy_locked();
    void count_token_stats_locked(const llama_moe_stream_layer & sl, const int32_t * ids, uint32_t n_ids, int64_t n_tokens);
    int32_t pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const;
    void reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot, bool update_policy = true);
    void refresh_expert_map_locked(llama_moe_stream_layer & sl) const;
    bool build_plan_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n);
    void apply_plan_locked(
            std::unique_lock<std::mutex> & lk,
            llama_moe_stream_layer & sl,
            size_t n_required = SIZE_MAX,
            bool update_policy = true,
            int64_t * resident_wait_us = nullptr);
    void commit_plan_locked(
            llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out, int64_t n);

    // multi-pass prefill helpers (called by llama_moe_stream_wave_ids, all under mtx)
    void plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n); // wave 0: build the plan
    void stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int32_t w, uint32_t n_ids); // make wave w resident + preload next
    void emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out, int32_t w, uint32_t n_ids, int64_t n_tok); // write the slot ids
};

// callback of the id-remapping custom op inserted by build_moe_ffn
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// callbacks of the multi-pass prefill custom ops inserted by build_moe_ffn when a ubatch touches
// more experts than the cache holds; each src[0] is the contiguous selected ids
//   wave_ids:  makes wave w's expert slice resident and emits slot ids (masked pairs park on a pool)
//   wave_mask: emits 1.0 for pairs belonging to wave w, 0.0 otherwise
void llama_moe_stream_wave_ids (ggml_tensor * dst, int ith, int nth, void * userdata);
void llama_moe_stream_wave_mask(ggml_tensor * dst, int ith, int nth, void * userdata);
