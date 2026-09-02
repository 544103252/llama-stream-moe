#include "llama-moe-stream.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

static const uint32_t MOE_STREAM_IO_THREADS_DEFAULT = 9;
static const uint32_t MOE_STREAM_IO_THREADS_MAX     = 18;
static const int64_t  MOE_STREAM_HOT_DECAY_TOKENS   = 64;

// O_DIRECT alignment: 4096 is a multiple of any device logical block size (512/4096), so it is
// universally valid, and reading a few extra KB of head/tail padding per slab is negligible
static const size_t MOE_STREAM_DIRECT_ALIGN = 4096;

// saturating increment - route-hotness counters accumulate over a whole run and must not wrap
static uint32_t sat_inc(uint32_t & c) {
    if (c < UINT32_MAX - 1) {
        c++;
    }
    return c;
}

// page-aligned allocation, required both for O_DIRECT reads and for Metal private-buffer uploads
static void * moe_aligned_alloc(size_t n) {
#ifdef _WIN32
    return _aligned_malloc(n, MOE_STREAM_DIRECT_ALIGN);
#else
    void * p = nullptr;
    if (posix_memalign(&p, MOE_STREAM_DIRECT_ALIGN, n) != 0) {
        p = nullptr;
    }
    return p;
#endif
}

static void moe_aligned_free(void * p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

// read len bytes at file offset offs into staging (thread-safe positional read); staging must have
// room for len (+ 2*MOE_STREAM_DIRECT_ALIGN when direct). returns a pointer to the len bytes
// within staging, or nullptr on failure
static const uint8_t * llama_moe_stream_pread(llama_file & file, uint8_t * staging, size_t len, size_t offs, bool direct) {
#ifdef _WIN32
    GGML_UNUSED(direct);
    // no positional read primitive; serialize the seek+read pairs
    static std::mutex io_mtx;
    std::lock_guard<std::mutex> lock(io_mtx);
    try {
        file.seek(offs, SEEK_SET);
        file.read_raw(staging, len);
        return staging;
    } catch (...) {
        return nullptr;
    }
#else
    const int fd = file.file_id();

    if (direct) {
        // O_DIRECT requires the offset, length, and buffer all block-aligned
        const size_t a     = MOE_STREAM_DIRECT_ALIGN;
        const size_t aoffs = offs & ~(a - 1);
        const size_t head  = offs - aoffs;
        const size_t total = ((head + len + a - 1)/a)*a;
        ssize_t r;
        do {
            r = pread(fd, staging, total, aoffs);
        } while (r < 0 && errno == EINTR);
        if (r < 0 || (size_t) r < head + len) {
            return nullptr;
        }
        return staging + head;
    }

    uint8_t * p    = staging;
    size_t    left = len;
    while (left > 0) {
        const ssize_t r = pread(fd, p, left, offs);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return nullptr;
        }
        if (r == 0) {
            return nullptr; // unexpected EOF
        }
        p    += r;
        offs += (size_t) r;
        left -= (size_t) r;
    }
    return staging;
#endif
}

// true iff all of the given exps tensors are this layer's cache tensors - guards against a second,
// non-streamed expert group on the same layer index (e.g. grovemoe chexps)
bool llama_moe_stream_layer::matches(const ggml_tensor * gate, const ggml_tensor * up,
                                     const ggml_tensor * down, const ggml_tensor * gate_up) const {
    auto is_cache = [this](const ggml_tensor * t) {
        for (const auto & w : weights) {
            if (w.cache == t) {
                return true;
            }
        }
        return false;
    };

    size_t n = 0;
    for (const ggml_tensor * t : { gate, up, down, gate_up }) {
        if (t == nullptr) {
            continue;
        }
        if (!is_cache(t)) {
            return false;
        }
        n++;
    }

    return n > 0 && n == weights.size();
}

// sizes the per-layer table and clamps the I/O thread count; workers are spawned lazily on first use
llama_moe_stream::llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct) : n_slots(n_slots) {
    layers.resize(n_layer);

    this->n_io_threads = n_io_threads <= 0 ? MOE_STREAM_IO_THREADS_DEFAULT : n_io_threads;
    this->n_io_threads = std::min<int32_t>(this->n_io_threads, MOE_STREAM_IO_THREADS_MAX);

    debug         = std::getenv("LLAMA_MOE_STREAM_DEBUG") != nullptr;
    shadow        = std::getenv("LLAMA_MOE_STREAM_SHADOW") != nullptr;
    gpu_decode_requested = std::getenv("LLAMA_MOE_STREAM_GPU_DECODE") != nullptr;
    gpu_decode_continuous_requested = std::getenv("LLAMA_MOE_STREAM_GPU_DECODE_CONTINUOUS") != nullptr;
    use_direct_io = direct;
}

// stop and join the I/O workers before the cache buffers and files they use are destroyed
llama_moe_stream::~llama_moe_stream() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        shutting_down = true;
        q_demand.clear();
    }
    cv_work.notify_all();
    for (auto & w : workers) {
        w.join();
    }
}

ggml_tensor * llama_moe_stream::create_cache_tensor(
        int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
        uint16_t file_idx, size_t offs) {
    GGML_ASSERT(il >= 0 && (size_t) il < layers.size());
    GGML_ASSERT(ggml_is_contiguous(meta));
    GGML_ASSERT(meta->ne[2] > 0 && meta->ne[3] == 1);

    const uint32_t n_expert  = meta->ne[2];
    const size_t   nb_expert = ggml_nbytes(meta) / n_expert;
    GGML_ASSERT(nb_expert * n_expert == ggml_nbytes(meta));
    GGML_ASSERT(n_slots > 0 && n_slots < n_expert);

    ggml_context * ctx = nullptr;
    for (auto & [cur_buft, cur_ctx] : ctxs) {
        if (cur_buft == buft) {
            ctx = cur_ctx.get();
            break;
        }
    }
    if (ctx == nullptr) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(layers.size()*4 + 1),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create ggml context for MoE expert streaming");
        }
        ctxs.emplace_back(buft, ctx);
    }

    ggml_tensor * cache = ggml_new_tensor_3d(ctx, meta->type, meta->ne[0], meta->ne[1], n_slots);
    ggml_format_name(cache, "%s.stream_cache", meta->name);
    GGML_ASSERT(ggml_nbytes(cache) == nb_expert * n_slots);

    auto & sl = layers[il];
    if (!sl) {
        sl = std::make_unique<llama_moe_stream_layer>();
        sl->mgr      = this;
        sl->il       = il;
        sl->n_expert = n_expert;
        sl->n_slots  = n_slots;
        sl->slot_expert  .resize(n_slots, -1);
        sl->slot_state   .resize(n_slots, LLAMA_MOE_STREAM_SLOT_EMPTY);
        sl->slot_claimed .resize(n_slots, 0);
        sl->slot_gen     .resize(n_slots, 0);
        sl->slot_last_use.resize(n_slots, 0);
        sl->expert_map   .resize(n_expert, -1);
        sl->route_hotness.resize(n_expert, 0);
        sl->seen         .resize(n_expert, 0);
        sl->keep         .resize(n_slots, 0);
    }
    GGML_ASSERT(sl->n_expert == n_expert);

    sl->weights.push_back({ cache, file_idx, offs, nb_expert });

    max_nb_expert = std::max(max_nb_expert, nb_expert);

    return cache;
}

void llama_moe_stream::alloc_bufs(bool no_alloc) {
    for (auto & [buft, ctx_ptr] : ctxs) {
        ggml_context * ctx = ctx_ptr.get();
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        ggml_backend_buffer_t buf;
        if (no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
        if (buf == nullptr) {
            throw std::runtime_error(format("unable to allocate %s buffer for MoE expert streaming", ggml_backend_buft_name(buft)));
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bufs.emplace_back(buf);

        LLAMA_LOG_INFO("%s: %12s expert cache size = %8.2f MiB (%u slots per layer)\n",
                __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0, n_slots);
    }

    if (!shadow || no_alloc) {
        return;
    }

    bool enabled = false;
    for (auto & layer_ptr : layers) {
        if (!layer_ptr || layer_ptr->weights.empty()) {
            continue;
        }
        auto & sl = *layer_ptr;
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(
                ggml_backend_buffer_get_type(sl.weights.front().cache->buffer));
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        auto get_ops = reg ? (ggml_backend_get_moe_stream_cache_ops_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_moe_stream_cache_ops") : nullptr;
        if (get_ops == nullptr) {
            continue;
        }

        ggml_backend_t backend = nullptr;
        for (const auto & owned : shadow_backends) {
            if (ggml_backend_get_device(owned.get()) == dev) {
                backend = owned.get();
                break;
            }
        }
        if (backend == nullptr) {
            backend = ggml_backend_dev_init(dev, nullptr);
            if (backend == nullptr) {
                continue;
            }
            shadow_backends.emplace_back(backend);
        }

        sl.shadow_backend = backend;
        sl.shadow_ops = get_ops();
        sl.shadow_available_map.resize(sl.n_expert, -1);
        enabled = true;
    }

    shadow = enabled;
    if (shadow) {
        LLAMA_LOG_INFO("%s: Stream MoE GPU planner shadow verification enabled\n", __func__);
    } else {
        LLAMA_LOG_WARN("%s: LLAMA_MOE_STREAM_SHADOW requested, but no supported backend was found\n", __func__);
    }
}

void llama_moe_stream::bind_decode_backends(const std::vector<ggml_backend_t> & backends) {
    unbind_decode_backends();
    if (!gpu_decode_requested) {
        return;
    }

    ggml_backend_t common_backend = nullptr;
    bool enabled = true;
    for (auto & layer_ptr : layers) {
        if (!layer_ptr || layer_ptr->weights.empty()) {
            continue;
        }
        auto & sl = *layer_ptr;
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(
                ggml_backend_buffer_get_type(sl.weights.front().cache->buffer));
        ggml_backend_t backend = nullptr;
        for (ggml_backend_t candidate : backends) {
            if (ggml_backend_get_device(candidate) == dev) {
                backend = candidate;
                break;
            }
        }

        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        auto get_ops = reg ? (ggml_backend_get_moe_stream_cache_ops_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_moe_stream_cache_ops") : nullptr;
        if (backend == nullptr || get_ops == nullptr ||
                (common_backend != nullptr && common_backend != backend)) {
            enabled = false;
            break;
        }

        common_backend = backend;
        sl.decode_backend = backend;
        sl.decode_ops = get_ops();
        sl.decode_available_map.resize(sl.n_expert, -1);
    }

    gpu_decode = enabled && common_backend != nullptr;
    gpu_decode_continuous = gpu_decode && gpu_decode_continuous_requested;
    if (gpu_decode_continuous) {
        for (const auto & layer_ptr : layers) {
            if (layer_ptr && layer_ptr->decode_ops != nullptr &&
                    (layer_ptr->decode_ops->supports_continuous == nullptr ||
                     !layer_ptr->decode_ops->supports_continuous(layer_ptr->decode_backend) ||
                     layer_ptr->decode_ops->continuous_begin == nullptr ||
                     layer_ptr->decode_ops->continuous_wait == nullptr ||
                     layer_ptr->decode_ops->continuous_resume == nullptr ||
                     layer_ptr->decode_ops->continuous_status == nullptr ||
                     layer_ptr->decode_ops->continuous_end == nullptr)) {
                gpu_decode_continuous = false;
                break;
            }
        }
    }
    if (gpu_decode) {
        LLAMA_LOG_INFO("%s: Stream MoE GPU decode planner enabled\n", __func__);
        if (gpu_decode_continuous) {
            LLAMA_LOG_INFO("%s: Stream MoE continuous GPU decode enabled\n", __func__);
        } else if (gpu_decode_continuous_requested) {
            LLAMA_LOG_WARN("%s: continuous GPU decode requested, but the backend does not support it\n", __func__);
        }
    } else {
        unbind_decode_backends();
        LLAMA_LOG_WARN("%s: LLAMA_MOE_STREAM_GPU_DECODE requested, but the streamed layers do not share one supported backend\n",
                __func__);
    }
}

void llama_moe_stream::unbind_decode_backends() {
    gpu_decode = false;
    gpu_decode_continuous = false;
    gpu_decode_state_ready = false;
    gpu_decode_cpu_policy_stale = false;
    for (auto & layer_ptr : layers) {
        if (!layer_ptr) {
            continue;
        }
        layer_ptr->decode_backend = nullptr;
        layer_ptr->decode_ops = nullptr;
        layer_ptr->decode_available_map.clear();
    }
}

void llama_moe_stream::record_continuous_hits(size_t n_plans) {
    if (n_plans == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mtx);
    stats.n_calls += n_plans;
    gpu_decode_cpu_policy_stale = true;
    if (token_stats_active) {
        token_stats.n_hit += n_plans;
    }
}

bool llama_moe_stream::prepare_decode() {
    if (!gpu_decode) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mtx);
    if (gpu_decode_state_ready) {
        return true;
    }
    if (!sync_decode_policy_locked()) {
        return false;
    }

    for (auto & layer_ptr : layers) {
        if (!layer_ptr) {
            continue;
        }
        auto & sl = *layer_ptr;
        if (sl.decode_backend == nullptr || sl.decode_ops == nullptr) {
            return false;
        }

        refresh_expert_map_locked(sl);
        std::fill(sl.decode_available_map.begin(), sl.decode_available_map.end(), -1);
        for (const auto & entry : sl.expert_slot) {
            sl.decode_available_map[entry.first] = entry.second;
        }

        const ggml_backend_moe_stream_cache_state state = {
            /* .layer              = */ sl.il,
            /* .n_expert           = */ sl.n_expert,
            /* .n_slots            = */ sl.n_slots,
            /* .expert_map         = */ sl.expert_map.data(),
            /* .available_map      = */ sl.decode_available_map.data(),
            /* .slot_expert        = */ sl.slot_expert.data(),
            /* .slot_state         = */ sl.slot_state.data(),
            /* .route_hotness      = */ sl.route_hotness.data(),
            /* .slot_last_use      = */ sl.slot_last_use.data(),
            /* .use_counter        = */ (uint64_t) sl.use_counter,
            /* .n_calls            = */ stats.n_calls,
            /* .hot_decay_interval = */ hot_decay_interval,
        };
        if (!sl.decode_ops->sync(sl.decode_backend, &state)) {
            LLAMA_LOG_ERROR("%s: failed to initialize GPU decode state for layer %d\n", __func__, sl.il);
            return false;
        }
    }

    gpu_decode_state_ready = true;
    gpu_decode_cpu_policy_stale = false;
    return true;
}

bool llama_moe_stream::sync_decode_policy_locked() {
    if (!gpu_decode_cpu_policy_stale) {
        return true;
    }

    for (auto & layer_ptr : layers) {
        if (!layer_ptr) {
            continue;
        }
        auto & sl = *layer_ptr;
        if (sl.decode_backend == nullptr || sl.decode_ops == nullptr || sl.decode_ops->read_policy == nullptr) {
            return false;
        }

        uint64_t use_counter = 0;
        if (!sl.decode_ops->read_policy(
                    sl.decode_backend,
                    sl.il,
                    sl.route_hotness.data(),
                    sl.route_hotness.size(),
                    sl.slot_last_use.data(),
                    sl.slot_last_use.size(),
                    &use_counter)) {
            LLAMA_LOG_ERROR("%s: failed to read GPU decode policy for layer %d\n", __func__, sl.il);
            return false;
        }
        sl.use_counter = (int64_t) use_counter;
    }

    gpu_decode_cpu_policy_stale = false;
    return true;
}

void llama_moe_stream::open_files(const std::vector<std::string> & paths) {
    for (const auto & path : paths) {
        if (path.empty()) {
            throw std::runtime_error("MoE expert streaming requires a file-based model (not a stream/file descriptor)");
        }
    }

    auto open_all = [&](bool direct) {
        files.clear();
        for (const auto & path : paths) {
            files.emplace_back(new llama_file(path.c_str(), "rb", direct));
        }
    };

    open_all(use_direct_io);

    // fall back to buffered when O_DIRECT is unusable: either the open did not honor it (macOS,
    // Windows, unsupported filesystems), or it opened but a probe read fails (some network/overlay
    // filesystems accept the flag then reject aligned reads). reopening is needed because O_DIRECT
    // is a property of the fd. done here, single-threaded, before any worker starts.
    if (use_direct_io) {
        bool ok = !files.empty() && files.front()->has_direct_io();
        if (ok) {
            uint8_t * probe = (uint8_t *) moe_aligned_alloc(MOE_STREAM_DIRECT_ALIGN);
            GGML_ASSERT(probe != nullptr);
            ok = llama_moe_stream_pread(*files.front(), probe, MOE_STREAM_DIRECT_ALIGN, 0, /*direct =*/ true) != nullptr;
            moe_aligned_free(probe);
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: O_DIRECT not usable, falling back to buffered streaming reads\n", __func__);
            use_direct_io = false;
            open_all(false);
        }
    }

    if (use_direct_io) {
        LLAMA_LOG_INFO("%s: MoE expert streaming uses O_DIRECT (page cache bypassed)\n", __func__);
    }

    // one token drives ~one remap per streamed layer, so decaying every 64 tokens is
    //   64 * n_streamed_layers remap calls (computed once here, off the hot path)
    int64_t n_streamed = 0;
    for (const auto & sl : layers) {
        n_streamed += sl != nullptr;
    }
    hot_decay_interval = MOE_STREAM_HOT_DECAY_TOKENS * n_streamed;
}

// spawn the I/O thread pool on first use (from the remap callback, under mtx)
void llama_moe_stream::start_workers_locked() {
    if (workers_started) {
        return;
    }
    workers_started = true;
    workers.reserve(n_io_threads);
    for (int32_t i = 0; i < n_io_threads; i++) {
        workers.emplace_back([this]() { worker_loop(); });
    }
}

// I/O worker: pops a reserved load, reads its expert slab(s) from the GGUF file into the cache
// slot, and marks the slot RESIDENT (or flags load_failed); stale/duplicate items are skipped
void llama_moe_stream::worker_loop() {
    // page-aligned staging (Metal private buffers require page-aligned source + page-multiple
    // length; O_DIRECT needs the extra head/tail slack for its aligned reads)
    uint8_t * staging = (uint8_t *) moe_aligned_alloc(max_nb_expert + 2*MOE_STREAM_DIRECT_ALIGN);
    GGML_ASSERT(staging != nullptr);

    std::unique_lock<std::mutex> lk(mtx);
    while (true) {
        cv_work.wait(lk, [&]{ return shutting_down || !q_demand.empty(); });
        if (shutting_down) {
            break;
        }

        llama_moe_stream_work w = q_demand.front();
        q_demand.pop_front();

        auto & sl = *w.sl;
        if (w.gen != sl.slot_gen[w.slot] ||
            sl.slot_state[w.slot] != LLAMA_MOE_STREAM_SLOT_LOADING ||
            sl.slot_expert[w.slot] != w.expert ||
            sl.slot_claimed[w.slot]) {
            continue; // stale or duplicate item
        }
        sl.slot_claimed[w.slot] = 1;

        const int64_t queue_us = w.queued_us > 0 ? ggml_time_us() - w.queued_us : 0;
        int64_t read_us = 0;
        int64_t upload_us = 0;

        lk.unlock();

        bool ok = true;
        for (const auto & wt : sl.weights) {
            const int64_t read_start_us = ggml_time_us();
            const uint8_t * data = llama_moe_stream_pread(*files[wt.file_idx], staging, wt.nb_expert, wt.offs + (size_t) w.expert*wt.nb_expert, use_direct_io);
            read_us += ggml_time_us() - read_start_us;
            if (data == nullptr) {
                ok = false;
                break;
            }
            const int64_t upload_start_us = ggml_time_us();
            ggml_backend_tensor_set(wt.cache, data, (size_t) w.slot*wt.nb_expert, wt.nb_expert);
            upload_us += ggml_time_us() - upload_start_us;
        }

        lk.lock();

        if (token_stats_active) {
            token_stats.n_worker_loads++;
            token_stats.t_worker_queue_us += queue_us;
            token_stats.t_worker_read_us += read_us;
            token_stats.t_worker_upload_us += upload_us;
        }

        sl.slot_claimed[w.slot] = 0;
        if (!ok) {
            load_failed = true;
        } else {
            sl.slot_state[w.slot] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
        }
        cv_done.notify_all();
    }
    lk.unlock();

    moe_aligned_free(staging);
}

// least valuable evictable slot: empty first, then coldest resident (min route hotness, oldest use
// as tiebreak); LOADING and keep slots are never candidates. returns -1 when no candidate exists
int32_t llama_moe_stream::pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const {
    int32_t v = -1;

    for (uint32_t s = 0; s < sl.n_slots; s++) {
        if ((keep && keep[s]) || sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
            continue;
        }
        if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_EMPTY) {
            return s;
        }
        if (v < 0) {
            v = s;
            continue;
        }
        const uint32_t hs = sl.route_hotness[sl.slot_expert[s]];
        const uint32_t hv = sl.route_hotness[sl.slot_expert[v]];
        if (hs < hv || (hs == hv && sl.slot_last_use[s] < sl.slot_last_use[v])) {
            v = s;
        }
    }

    return v;
}

// bind expert -> slot and mark it LOADING: evict the slot's prior occupant, bump slot_gen (so any
// in-flight load for the old occupant is recognized as stale), and update the expert_slot index
void llama_moe_stream::reserve_slot_locked(
        llama_moe_stream_layer & sl, int32_t expert, int32_t slot, bool update_policy) {
    if (sl.slot_expert[slot] >= 0) {
        if (debug) {
            LLAMA_LOG_DEBUG("%s: layer %d: evict expert %d from slot %d\n", __func__, sl.il, sl.slot_expert[slot], slot);
        }
        sl.expert_slot.erase(sl.slot_expert[slot]);
    }

    sl.slot_expert[slot] = expert;
    sl.slot_state[slot]  = LLAMA_MOE_STREAM_SLOT_LOADING;
    sl.slot_gen[slot]++;
    if (update_policy) {
        sl.slot_last_use[slot] = ++sl.use_counter;
    }
    sl.expert_slot[expert] = slot;
    sl.seen[expert] = 1;
}

void llama_moe_stream::refresh_expert_map_locked(llama_moe_stream_layer & sl) const {
    std::fill(sl.expert_map.begin(), sl.expert_map.end(), -1);
    for (uint32_t s = 0; s < sl.n_slots; s++) {
        if (sl.slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
            continue;
        }
        const int32_t expert = sl.slot_expert[s];
        GGML_ASSERT(expert >= 0 && (uint32_t) expert < sl.n_expert);
        GGML_ASSERT(sl.expert_map[expert] == -1);
        sl.expert_map[expert] = s;
    }
}

bool llama_moe_stream::build_plan_locked(
        llama_moe_stream_layer & sl, const int32_t * ids, int64_t n) {
    auto & plan = sl.plan;

    plan.loads.clear();
    plan.next_map.clear();
    plan.mapped_topk.resize(n);
    plan.required_slots.clear();
    std::fill(sl.keep.begin(), sl.keep.end(), 0);

    refresh_expert_map_locked(sl);

    bool has_missing = false;
    for (const int32_t expert : sl.uniq) {
        if (sl.expert_slot.find(expert) == sl.expert_slot.end()) {
            has_missing = true;
            break;
        }
    }

    if (!has_missing) {
        for (const int32_t expert : sl.uniq) {
            const int32_t slot = sl.expert_slot.at(expert);
            sl.keep[slot] = 1;
            plan.required_slots.push_back(slot);
        }
        for (int64_t i = 0; i < n; i++) {
            plan.mapped_topk[i] = sl.expert_slot.at(ids[i]);
        }
        return true;
    }

    plan.next_map = sl.expert_map;
    for (const int32_t expert : sl.uniq) {
        const auto it = sl.expert_slot.find(expert);
        if (it != sl.expert_slot.end() &&
                sl.slot_state[it->second] == LLAMA_MOE_STREAM_SLOT_LOADING) {
            plan.next_map[expert] = it->second;
        }
    }

    for (const int32_t expert : sl.uniq) {
        int32_t slot = plan.next_map[expert];
        if (slot < 0) {
            slot = pick_victim_locked(sl, sl.keep.data());
            if (slot < 0) {
                return false;
            }

            const int32_t victim = sl.slot_state[slot] == LLAMA_MOE_STREAM_SLOT_EMPTY ?
                    -1 : sl.slot_expert[slot];
            GGML_ASSERT(victim < 0 || plan.next_map[victim] == slot);

            plan.loads.push_back({ expert, victim, slot });
            if (victim >= 0) {
                plan.next_map[victim] = -1;
            }
            plan.next_map[expert] = slot;
        }

        sl.keep[slot] = 1;
        plan.required_slots.push_back(slot);
    }

    for (int64_t i = 0; i < n; i++) {
        const int32_t slot = plan.next_map[ids[i]];
        GGML_ASSERT(slot >= 0 && (uint32_t) slot < sl.n_slots);
        plan.mapped_topk[i] = slot;
    }

    return true;
}

void llama_moe_stream::apply_plan_locked(
        std::unique_lock<std::mutex> & lk,
        llama_moe_stream_layer & sl,
        size_t n_required,
        bool update_policy,
        int64_t * resident_wait_us) {
    auto & plan = sl.plan;
    if (resident_wait_us != nullptr) {
        *resident_wait_us = 0;
    }
    if (n_required == SIZE_MAX) {
        n_required = sl.uniq.size();
    }
    GGML_ASSERT(n_required >= plan.loads.size());

    for (const auto & load : plan.loads) {
        GGML_ASSERT(load.expert >= 0 && (uint32_t) load.expert < sl.n_expert);
        GGML_ASSERT(load.slot >= 0 && (uint32_t) load.slot < sl.n_slots);
        GGML_ASSERT(load.victim < 0 || sl.slot_expert[load.slot] == load.victim);

        if (!sl.seen[load.expert]) {
            stats.n_miss_cold++;
        }
        reserve_slot_locked(sl, load.expert, load.slot, update_policy);
    }

    stats.n_miss += plan.loads.size();
    stats.n_hit  += n_required - plan.loads.size();

    bool waited = false;
    for (const int32_t slot : plan.required_slots) {
        GGML_ASSERT(slot >= 0 && (uint32_t) slot < sl.n_slots);
        if (sl.slot_state[slot] == LLAMA_MOE_STREAM_SLOT_LOADING) {
            q_demand.push_back({ &sl, sl.slot_expert[slot], slot, sl.slot_gen[slot], ggml_time_us() });
            cv_work.notify_one();
            waited = true;
        }
    }

    if (!waited) {
        return;
    }

    const int64_t t0 = ggml_time_us();
    cv_done.wait(lk, [&] {
        if (load_failed) {
            return true;
        }
        for (const int32_t slot : plan.required_slots) {
            if (sl.slot_state[slot] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                return false;
            }
        }
        return true;
    });
    if (load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }
    const int64_t elapsed_us = ggml_time_us() - t0;
    stats.t_stall_us += elapsed_us;
    if (resident_wait_us != nullptr) {
        *resident_wait_us = elapsed_us;
    }
}

void llama_moe_stream::commit_plan_locked(
        llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out, int64_t n) {
    auto & plan = sl.plan;

    GGML_ASSERT(plan.mapped_topk.size() == (size_t) n);
    for (int64_t i = 0; i < n; i++) {
        const int32_t expert = ids[i];
        const int32_t slot = plan.mapped_topk[i];
        GGML_ASSERT(slot >= 0 && (uint32_t) slot < sl.n_slots);
        GGML_ASSERT(sl.slot_state[slot] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
        GGML_ASSERT(sl.slot_expert[slot] == expert);
        if (!plan.next_map.empty()) {
            GGML_ASSERT(plan.next_map[expert] == slot);
        }
    }

    if (!plan.next_map.empty()) {
        for (uint32_t expert = 0; expert < sl.n_expert; expert++) {
            const int32_t slot = plan.next_map[expert];
            if (slot >= 0) {
                GGML_ASSERT((uint32_t) slot < sl.n_slots);
                GGML_ASSERT(sl.slot_state[slot] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
                GGML_ASSERT(sl.slot_expert[slot] == (int32_t) expert);
            }
        }
        sl.expert_map = plan.next_map;
    } else {
        for (int64_t i = 0; i < n; i++) {
            sl.expert_map[ids[i]] = plan.mapped_topk[i];
        }
    }

    for (int64_t i = 0; i < n; i++) {
        const int32_t slot = plan.mapped_topk[i];
        sl.slot_last_use[slot] = ++sl.use_counter;
        out[i] = slot;
    }
}

size_t llama_moe_stream::size_bufs() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }
    return size;
}

void llama_moe_stream::token_stats_begin() {
    std::lock_guard<std::mutex> lock(mtx);
    token_stats = {};
    token_stats_active = true;
}

llama_moe_stream_token_stats llama_moe_stream::token_stats_end() {
    std::lock_guard<std::mutex> lock(mtx);
    token_stats_active = false;
    return token_stats;
}

void llama_moe_stream::count_token_stats_locked(
        const llama_moe_stream_layer & sl, const int32_t * ids, uint32_t n_ids, int64_t n_tokens) {
    if (!token_stats_active) {
        return;
    }
    for (int64_t t = 0; t < n_tokens; t++) {
        bool hit = true;
        for (uint32_t r = 0; r < n_ids; r++) {
            const int32_t e = ids[t*n_ids + r];
            const auto it = sl.expert_slot.find(e);
            if (it == sl.expert_slot.end() || sl.slot_state[it->second] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                hit = false;
                break;
            }
        }
        if (hit) {
            token_stats.n_hit++;
        } else {
            token_stats.n_miss++;
        }
    }
}

void llama_moe_stream::print_stats() const {
    std::lock_guard<std::mutex> lock(mtx);

    const int64_t n_touched = stats.n_hit + stats.n_miss;
    LLAMA_LOG_INFO("%s: moe stream: remap calls = %" PRId64 ", expert hits = %" PRId64 ", misses = %" PRId64 " (%" PRId64 " cold), hit rate = %.2f%%\n",
            __func__, stats.n_calls, stats.n_hit, stats.n_miss, stats.n_miss_cold,
            n_touched > 0 ? 100.0*stats.n_hit/n_touched : 0.0);
    LLAMA_LOG_INFO("%s: moe stream: load stall = %.2f ms total (%.3f ms per remap call)\n",
            __func__, stats.t_stall_us/1000.0, stats.n_calls > 0 ? stats.t_stall_us/1000.0/stats.n_calls : 0.0);
    if (stats.n_wave_calls > 0) {
        LLAMA_LOG_INFO("%s: moe stream: waves = %" PRId64 " (%" PRId64 " non-empty), preloads issued = %" PRId64 " (ready on arrival = %" PRId64 "), wave stall = %.2f ms\n",
                __func__, stats.n_wave_calls, stats.n_waves_run, stats.n_preload_issued, stats.n_preload_ready, stats.t_stall_wave_us/1000.0);
    }
    if (stats.n_shadow_plans > 0) {
        LLAMA_LOG_INFO("%s: moe stream shadow: plans = %" PRId64 ", matched = %" PRId64 ", mismatched = %" PRId64 "\n",
                __func__, stats.n_shadow_plans, stats.n_shadow_plans - stats.n_shadow_mismatches,
                stats.n_shadow_mismatches);
    }
}

static bool llama_moe_stream_shadow_prepare(
        llama_moe_stream_layer & sl,
        const ggml_tensor * selected) {
    auto * mgr = sl.mgr;
    if (!mgr->shadow || sl.shadow_backend == nullptr || sl.shadow_ops == nullptr) {
        return false;
    }

    std::fill(sl.shadow_available_map.begin(), sl.shadow_available_map.end(), -1);
    for (const auto & entry : sl.expert_slot) {
        sl.shadow_available_map[entry.first] = entry.second;
    }

    const ggml_backend_moe_stream_cache_state state = {
        /* .layer           = */ sl.il,
        /* .n_expert        = */ sl.n_expert,
        /* .n_slots         = */ sl.n_slots,
        /* .expert_map      = */ sl.expert_map.data(),
        /* .available_map   = */ sl.shadow_available_map.data(),
        /* .slot_expert     = */ sl.slot_expert.data(),
        /* .slot_state      = */ sl.slot_state.data(),
        /* .route_hotness   = */ sl.route_hotness.data(),
        /* .slot_last_use   = */ sl.slot_last_use.data(),
        /* .use_counter     = */ (uint64_t) sl.use_counter,
        /* .n_calls         = */ mgr->stats.n_calls,
        /* .hot_decay_interval = */ mgr->hot_decay_interval,
    };

    mgr->stats.n_shadow_plans++;
    if (mgr->token_stats_active) {
        mgr->token_stats.n_shadow_plans++;
    }
    if (!sl.shadow_ops->sync(sl.shadow_backend, &state)) {
        mgr->stats.n_shadow_mismatches++;
        if (mgr->token_stats_active) {
            mgr->token_stats.n_shadow_mismatches++;
        }
        LLAMA_LOG_ERROR("%s: layer %d: GPU state synchronization failed\n", __func__, sl.il);
        return false;
    }

    ggml_backend_moe_stream_cache_plan gpu = {};
    if (!sl.shadow_ops->prepare(sl.shadow_backend, selected, &gpu)) {
        mgr->stats.n_shadow_mismatches++;
        if (mgr->token_stats_active) {
            mgr->token_stats.n_shadow_mismatches++;
        }
        LLAMA_LOG_ERROR("%s: layer %d: GPU prepare failed\n", __func__, sl.il);
        sl.shadow_ops->abort(sl.shadow_backend, sl.il);
        return false;
    }

    const auto & cpu = sl.plan;
    const char * mismatch = nullptr;
    size_t mismatch_index = 0;
    int32_t cpu_value = 0;
    int32_t gpu_value = 0;

    if (gpu.n_loads != cpu.loads.size()) {
        mismatch = "n_loads";
        cpu_value = (int32_t) cpu.loads.size();
        gpu_value = (int32_t) gpu.n_loads;
    } else {
        for (size_t i = 0; i < cpu.loads.size() && mismatch == nullptr; ++i) {
            const int32_t cpu_load[3] = { cpu.loads[i].expert, cpu.loads[i].victim, cpu.loads[i].slot };
            const int32_t gpu_load[3] = { gpu.loads[i].expert, gpu.loads[i].victim, gpu.loads[i].slot };
            for (size_t field = 0; field < 3; ++field) {
                if (cpu_load[field] != gpu_load[field]) {
                    mismatch = field == 0 ? "loads.expert" : field == 1 ? "loads.victim" : "loads.slot";
                    mismatch_index = i;
                    cpu_value = cpu_load[field];
                    gpu_value = gpu_load[field];
                    break;
                }
            }
        }
    }

    if (mismatch == nullptr && gpu.n_mapped_topk != cpu.mapped_topk.size()) {
        mismatch = "mapped_topk.size";
        cpu_value = (int32_t) cpu.mapped_topk.size();
        gpu_value = (int32_t) gpu.n_mapped_topk;
    }
    for (size_t i = 0; i < cpu.mapped_topk.size() && mismatch == nullptr; ++i) {
        if (cpu.mapped_topk[i] != gpu.mapped_topk[i]) {
            mismatch = "mapped_topk";
            mismatch_index = i;
            cpu_value = cpu.mapped_topk[i];
            gpu_value = gpu.mapped_topk[i];
        }
    }

    if (mismatch == nullptr && gpu.n_next_map != cpu.next_map.size()) {
        mismatch = "next_map.size";
        cpu_value = (int32_t) cpu.next_map.size();
        gpu_value = (int32_t) gpu.n_next_map;
    }
    for (size_t i = 0; i < cpu.next_map.size() && mismatch == nullptr; ++i) {
        if (cpu.next_map[i] != gpu.next_map[i]) {
            mismatch = "next_map";
            mismatch_index = i;
            cpu_value = cpu.next_map[i];
            gpu_value = gpu.next_map[i];
        }
    }

    if (mismatch != nullptr) {
        mgr->stats.n_shadow_mismatches++;
        if (mgr->token_stats_active) {
            mgr->token_stats.n_shadow_mismatches++;
        }
        LLAMA_LOG_ERROR("%s: layer %d: %s mismatch at %zu: cpu=%d gpu=%d\n",
                __func__, sl.il, mismatch, mismatch_index, cpu_value, gpu_value);
        sl.shadow_ops->abort(sl.shadow_backend, sl.il);
        return false;
    }

    return true;
}

bool llama_moe_stream::eval_callback(
        ggml_backend_sched_t sched,
        ggml_tensor * tensor,
        bool ask,
        bool & requested) {
    requested = false;
    if (!gpu_decode || tensor == nullptr || tensor->op != GGML_OP_MOE_STREAM_CACHE_DECIDE) {
        return true;
    }

    int32_t layer_id = -1;
    memcpy(&layer_id, tensor->op_params, sizeof(layer_id));
    llama_moe_stream_layer * sl = layer(layer_id);
    if (sl == nullptr || sl->decode_backend == nullptr || sl->decode_ops == nullptr) {
        LLAMA_LOG_ERROR("%s: invalid GPU decode layer %d\n", __func__, layer_id);
        gpu_decode_state_ready = false;
        return false;
    }

    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
    if (backend != sl->decode_backend) {
        LLAMA_LOG_ERROR("%s: layer %d cache planner was assigned to an unexpected backend\n",
                __func__, layer_id);
        gpu_decode_state_ready = false;
        return false;
    }

    requested = true;
    if (ask) {
        return true;
    }

    const int64_t callback_start_us = ggml_time_us();
    const int64_t sync_us = ggml_backend_sched_get_last_eval_callback_sync_us(sched);
    const int64_t segment_us = ggml_backend_sched_get_last_eval_callback_segment_us(sched);
    ggml_backend_moe_stream_cache_plan gpu = {};
    const int64_t prepare_start_us = ggml_time_us();
    if (!sl->decode_ops->prepare(backend, tensor, &gpu)) {
        LLAMA_LOG_ERROR("%s: GPU decode prepare failed for layer %d\n", __func__, layer_id);
        gpu_decode_state_ready = false;
        return false;
    }
    const int64_t prepare_us = ggml_time_us() - prepare_start_us;
    const auto abort_plan = [&] {
        sl->decode_ops->abort(backend, layer_id);
        gpu_decode_state_ready = false;
    };

    const bool slow_path = gpu.n_loads > 0 || gpu.n_waiting > 0;
    if (gpu.n_required == 0 || gpu.n_required > sl->n_slots ||
            gpu.n_loads > gpu.n_required || gpu.n_waiting > gpu.n_required ||
            (gpu.n_loads > 0 && gpu.loads == nullptr) ||
            (slow_path &&
             (gpu.required_slots == nullptr || gpu.n_required_slots != gpu.n_required))) {
        LLAMA_LOG_ERROR("%s: GPU decode returned an invalid plan for layer %d\n", __func__, layer_id);
        abort_plan();
        return false;
    }

    std::unique_lock<std::mutex> lk(mtx);
    if (load_failed) {
        LLAMA_LOG_ERROR("%s: expert loading previously failed\n", __func__);
        abort_plan();
        return false;
    }
    if (token_stats_active && gpu.n_gpu_commit_carry > 0) {
        token_stats.n_gpu_commit_carry += gpu.n_gpu_commit_carry;
        token_stats.t_gpu_commit_carry_ns += gpu.t_gpu_commit_carry_ns;
    }

    stats.n_calls++;
    gpu_decode_cpu_policy_stale = true;
    if (token_stats_active) {
        if (slow_path) {
            token_stats.n_miss++;
        } else {
            token_stats.n_hit++;
        }
    }

    auto & plan = sl->plan;
    plan.loads.clear();
    plan.next_map.clear();
    plan.mapped_topk.clear();
    plan.required_slots.clear();
    int64_t load_us = 0;
    int64_t resident_wait_us = 0;

    if (slow_path) {
        start_workers_locked();
        plan.loads.reserve(gpu.n_loads);
        for (size_t i = 0; i < gpu.n_loads; ++i) {
            if (gpu.loads[i].expert < 0 || (uint32_t) gpu.loads[i].expert >= sl->n_expert ||
                    gpu.loads[i].slot < 0 || (uint32_t) gpu.loads[i].slot >= sl->n_slots ||
                    gpu.loads[i].victim < -1 ||
                    (gpu.loads[i].victim >= 0 && (uint32_t) gpu.loads[i].victim >= sl->n_expert)) {
                LLAMA_LOG_ERROR("%s: GPU decode returned an invalid load for layer %d\n", __func__, layer_id);
                abort_plan();
                return false;
            }
            plan.loads.push_back({ gpu.loads[i].expert, gpu.loads[i].victim, gpu.loads[i].slot });
        }
        std::fill(sl->keep.begin(), sl->keep.end(), 0);
        for (size_t i = 0; i < gpu.n_required_slots; ++i) {
            const int32_t slot = gpu.required_slots[i];
            if (slot < 0 || (uint32_t) slot >= sl->n_slots) {
                LLAMA_LOG_ERROR("%s: GPU decode returned an invalid mapped slot for layer %d\n",
                        __func__, layer_id);
                abort_plan();
                return false;
            }
            if (!sl->keep[slot]) {
                sl->keep[slot] = 1;
                plan.required_slots.push_back(slot);
            }
        }
        if (plan.required_slots.size() != gpu.n_required) {
            LLAMA_LOG_ERROR("%s: GPU decode required slot count mismatch for layer %d\n", __func__, layer_id);
            abort_plan();
            return false;
        }
        const int64_t load_start_us = ggml_time_us();
        apply_plan_locked(lk, *sl, gpu.n_required, false, &resident_wait_us);
        load_us = ggml_time_us() - load_start_us;
    } else {
        stats.n_hit += gpu.n_required;
    }

    const int64_t commit_start_us = ggml_time_us();
    if (!sl->decode_ops->commit(backend, layer_id)) {
        LLAMA_LOG_ERROR("%s: GPU decode commit failed for layer %d\n", __func__, layer_id);
        abort_plan();
        return false;
    }
    const int64_t commit_us = ggml_time_us() - commit_start_us;
    if (!slow_path && token_stats_active) {
        token_stats.n_gpu_hit_plans++;
        token_stats.t_gpu_hit_segment_ns += gpu.t_gpu_segment_ns;
        token_stats.t_gpu_hit_planner_ns += gpu.t_gpu_planner_ns;
        token_stats.t_gpu_hit_wall_us += segment_us;
        token_stats.t_gpu_hit_sync_us += sync_us;
        token_stats.t_gpu_hit_cb_us += ggml_time_us() - callback_start_us;
        token_stats.t_gpu_hit_prepare_us += prepare_us;
        token_stats.t_gpu_hit_commit_us += commit_us;
    } else if (slow_path && token_stats_active) {
        token_stats.n_gpu_slow_plans++;
        token_stats.n_gpu_slow_loads += gpu.n_loads;
        token_stats.t_gpu_slow_segment_ns += gpu.t_gpu_segment_ns;
        token_stats.t_gpu_slow_planner_ns += gpu.t_gpu_planner_ns;
        token_stats.t_gpu_slow_wall_us += segment_us;
        token_stats.t_gpu_slow_sync_us += sync_us;
        token_stats.t_gpu_slow_cb_us += ggml_time_us() - callback_start_us;
        token_stats.t_gpu_slow_prepare_us += prepare_us;
        token_stats.t_gpu_slow_load_us += load_us;
        token_stats.t_gpu_slow_commit_us += commit_us;
        token_stats.n_gpu_slow_waiting += gpu.n_waiting;
        token_stats.t_gpu_slow_resident_wait_us += resident_wait_us;
    }
    return true;
}

// custom-op callback (single-threaded on ith 0): given the router's expert ids, ensure every touched
// expert is resident - reserving cache slots and demand-loading misses, stalling until they commit -
// then rewrite each id to its cache slot. this only relabels ids, so the same experts are computed
// in the same order; the result matches a non-streamed run (bit-exact when both paths use the same
// kernels, as on CUDA; a CPU build that repacks the non-streamed weights can differ in the last bits).
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * sl  = (llama_moe_stream_layer *) userdata;
    auto * mgr = sl->mgr;

    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_are_same_shape(a, dst));

    const int64_t n = ggml_nelements(a);

    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk(mgr->mtx);
    if (!mgr->sync_decode_policy_locked()) {
        GGML_ABORT("MoE expert streaming: failed to synchronize the GPU cache policy");
    }
    mgr->gpu_decode_state_ready = false;

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_calls++;
    mgr->start_workers_locked();

    mgr->count_token_stats_locked(*sl, ids, (uint32_t) a->ne[0], a->ne[1]);

    // distinct experts touched by this ubatch, in first-use order
    sl->touched.assign(sl->n_expert, 0);
    sl->uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl->n_expert);
        if (!sl->touched[e]) {
            sl->touched[e] = 1;
            sl->uniq.push_back(e);
        }
    }

    if (sl->uniq.size() > sl->n_slots) {
        GGML_ABORT("MoE expert streaming: layer %d needs %zu distinct experts but the cache has only %u slots; "
                   "increase --moe-stream-cache or reduce the ubatch size (-ub)",
                sl->il, sl->uniq.size(), sl->n_slots);
    }

    // route hotness for eviction; halved periodically so a formerly-hot expert ages out
    for (const int32_t e : sl->uniq) {
        sat_inc(sl->route_hotness[e]);
    }
    if (mgr->hot_decay_interval > 0 && mgr->stats.n_calls % mgr->hot_decay_interval == 0) {
        for (auto & sl2 : mgr->layers) {
            if (sl2) {
                for (auto & h : sl2->route_hotness) {
                    h >>= 1;
                }
            }
        }
    }

    while (!mgr->build_plan_locked(*sl, ids, n)) {
        mgr->cv_done.wait(lk);
        if (mgr->load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
    }

    const bool shadow_pending = llama_moe_stream_shadow_prepare(*sl, a);
    mgr->apply_plan_locked(lk, *sl);
    mgr->commit_plan_locked(*sl, ids, out, n);
    if (shadow_pending && !sl->shadow_ops->commit(sl->shadow_backend, sl->il)) {
        mgr->stats.n_shadow_mismatches++;
        if (mgr->token_stats_active) {
            mgr->token_stats.n_shadow_mismatches++;
        }
        LLAMA_LOG_ERROR("%s: layer %d: GPU commit failed\n", __func__, sl->il);
        sl->shadow_ops->abort(sl->shadow_backend, sl->il);
    }
}

// stable per-wave userdata; grows lazily and records the per-wave expert capacity (set at build)
llama_moe_stream_wave * llama_moe_stream_layer::wave_userdata(int32_t wave, uint32_t capacity) {
    GGML_ASSERT(capacity >= 1 && capacity <= n_slots);
    plan_capacity = capacity;
    while ((size_t) wave >= wave_ud.size()) {
        auto ud = std::make_unique<llama_moe_stream_wave>();
        ud->sl   = this;
        ud->wave = (int32_t) wave_ud.size();
        wave_ud.push_back(std::move(ud));
    }
    return wave_ud[wave].get();
}

// wave 0 of a ubatch: record the distinct touched experts (sl.uniq, first-use order) and split them
// into consecutive groups of plan_capacity, one group per wave (sl.expert_wave[e] = e's wave)
void llama_moe_stream::plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n) {
    if (!sync_decode_policy_locked()) {
        GGML_ABORT("MoE expert streaming: failed to synchronize the GPU cache policy");
    }
    gpu_decode_state_ready = false;
    stats.n_calls++;
    start_workers_locked();

    sl.touched.assign(sl.n_expert, 0);
    sl.uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl.n_expert);
        if (!sl.touched[e]) {
            sl.touched[e] = 1;
            sl.uniq.push_back(e);
        }
    }

    GGML_ASSERT(sl.plan_capacity > 0);
    sl.expert_wave.assign(sl.n_expert, 0xff);
    for (size_t i = 0; i < sl.uniq.size(); i++) {
        GGML_ASSERT(i/sl.plan_capacity < 0xff);
        sl.expert_wave[sl.uniq[i]] = (uint8_t) (i/sl.plan_capacity);
    }
    sl.plan_n_waves   = (uint32_t) ((sl.uniq.size() + sl.plan_capacity - 1)/sl.plan_capacity);
    sl.plan_next_wave = 0;
}

// make wave w's expert slice (uniq[w*cap .. +count)) resident, waiting for its loads, and best-effort
// preload the next wave so its loads overlap this wave's compute. leaves sl.demand_slots = this wave's
// slots and sl.plan_pool = the resident parking pool (>= n_ids slots) the emit draws masked pairs from
void llama_moe_stream::stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int32_t w, uint32_t n_ids) {
    const size_t first = (size_t) w*sl.plan_capacity;
    const size_t count = first < sl.uniq.size() ? std::min<size_t>(sl.plan_capacity, sl.uniq.size() - first) : 0;

    std::fill(sl.keep.begin(), sl.keep.end(), 0);
    sl.demand_slots.clear();

    // a small final wave has fewer than n_ids own slots; borrow the rest from the previous wave's
    //   pool so every token row has n_ids distinct resident parking slots for its masked pairs
    std::vector<int32_t> borrowed;
    if (count < n_ids) {
        GGML_ASSERT(sl.plan_pool.size() >= n_ids - count);
        for (size_t i = 0; i < n_ids - count; i++) {
            borrowed.push_back(sl.plan_pool[i]);
            sl.keep[sl.plan_pool[i]] = 1; // parking slots must survive this wave's loads
        }
    }

    // protect the next wave's already-resident experts so this wave's victims do not evict them
    const size_t nfirst = first + sl.plan_capacity;
    const size_t ncount = nfirst < sl.uniq.size() ? std::min<size_t>(sl.plan_capacity, sl.uniq.size() - nfirst) : 0;
    for (size_t i = nfirst; i < nfirst + ncount; i++) {
        const auto it = sl.expert_slot.find(sl.uniq[i]);
        if (it != sl.expert_slot.end()) {
            sl.keep[it->second] = 1;
        }
    }

    // reserve and demand-load this wave's experts (per-expert, same path as the decode remap)
    bool waited = false;
    if (count > 0) {
        stats.n_waves_run++;
        for (size_t i = first; i < first + count; i++) {
            const int32_t e  = sl.uniq[i];
            const auto    it = sl.expert_slot.find(e);
            if (it != sl.expert_slot.end()) {
                // already in the cache (resident, or still loading from the previous wave's preload)
                const int32_t s = it->second;
                if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                    q_demand.push_back({ &sl, e, s, sl.slot_gen[s], ggml_time_us() }); // promote to demand, wait for it
                    cv_work.notify_one();
                    waited = true;
                } else {
                    stats.n_preload_ready++; // resident from the previous wave's preload
                }
                stats.n_hit++;
                sl.keep[s] = 1;
                sl.demand_slots.push_back(s);
            } else {
                // miss: evict a non-kept slot and queue the load
                int32_t v;
                while ((v = pick_victim_locked(sl, sl.keep.data())) < 0) {
                    cv_done.wait(lk);
                    if (load_failed) {
                        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                    }
                }
                if (!sl.seen[e]) {
                    stats.n_miss_cold++;
                }
                reserve_slot_locked(sl, e, v);
                q_demand.push_back({ &sl, e, v, sl.slot_gen[v], ggml_time_us() });
                cv_work.notify_one();
                stats.n_miss++;
                waited = true;
                sl.keep[v] = 1;
                sl.demand_slots.push_back(v);
            }
        }
    }

    // best-effort preload of the next wave so its loads overlap this wave's compute; never waits,
    //   whatever cannot be reserved now simply becomes the next wave's demand load
    if (std::getenv("LLAMA_MOE_STREAM_NO_PRELOAD") == nullptr) {
        for (size_t i = nfirst; i < nfirst + ncount; i++) {
            const int32_t e = sl.uniq[i];
            if (sl.expert_slot.find(e) != sl.expert_slot.end()) {
                continue;
            }
            const int32_t v = pick_victim_locked(sl, sl.keep.data());
            if (v < 0) {
                continue;
            }
            if (!sl.seen[e]) {
                stats.n_miss_cold++;
            }
            reserve_slot_locked(sl, e, v);
            sl.keep[v] = 1;
            q_demand.push_back({ &sl, e, v, sl.slot_gen[v], ggml_time_us() });
            cv_work.notify_one();
            stats.n_preload_issued++;
        }
    }

    if (waited) {
        const int64_t t0 = ggml_time_us();
        cv_done.wait(lk, [&]{
            if (load_failed) {
                return true;
            }
            for (const int32_t s : sl.demand_slots) {
                if (sl.slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        stats.t_stall_wave_us += ggml_time_us() - t0;
    }

    // parking pool: this wave's own resident slots plus the borrowed ones (all keep-protected;
    //   the next same-layer reservation is ordered after this wave's GEMMs by the graph)
    sl.plan_pool = sl.demand_slots;
    sl.plan_pool.insert(sl.plan_pool.end(), borrowed.begin(), borrowed.end());
    GGML_ASSERT(sl.plan_pool.size() >= n_ids);
}

// write out[i] = the cache slot the GEMM should index for each (token, expert) pair of wave w, one
// token row at a time: pairs whose expert is in this wave get its real slot; the rest park on distinct
// resident pool slots (pool_used prevents a repeat within the row, required by the Metal kernel)
void llama_moe_stream::emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out,
        int32_t w, uint32_t n_ids, int64_t n_tok) {
    for (int64_t t = 0; t < n_tok; t++) {
        sl.pool_used.clear();

        // pass 1: pairs whose expert belongs to this wave -> that expert's real (resident) slot
        for (uint32_t kk = 0; kk < n_ids; kk++) {
            const int64_t i = t*n_ids + kk;
            const int32_t e = ids[i];
            GGML_ASSERT(sl.expert_wave[e] != 0xff);
            if (sl.expert_wave[e] == (uint8_t) w) {
                const int32_t s = sl.expert_slot.at(e);
                GGML_ASSERT(sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
                sl.slot_last_use[s] = ++sl.use_counter;
                out[i] = s;
                sl.pool_used.push_back(s);
            }
        }

        // pass 2: the remaining (masked) pairs -> the next pool slot not yet used in this row
        size_t pi = 0;
        for (uint32_t kk = 0; kk < n_ids; kk++) {
            const int64_t i = t*n_ids + kk;
            if (sl.expert_wave[ids[i]] == (uint8_t) w) {
                continue;
            }
            while (std::find(sl.pool_used.begin(), sl.pool_used.end(), sl.plan_pool[pi]) != sl.pool_used.end()) {
                pi++;
                GGML_ASSERT(pi < sl.plan_pool.size());
            }
            GGML_ASSERT(sl.slot_state[sl.plan_pool[pi]] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
            out[i] = sl.plan_pool[pi];
            sl.pool_used.push_back(sl.plan_pool[pi]);
            pi++;
        }
    }
}

// Custom-op callback for one pass of multi-pass prefill. When a ubatch touches more experts than the
// cache holds, build_moe_ffn runs the expert GEMMs in several waves; this runs once per wave (single-
// threaded on ith 0), in wave order. For wave w it makes that wave's expert slice resident (preloading
// the next wave), then writes the slot ids the GEMM indexes - see plan_waves_locked / stage_wave_locked
// / emit_wave_slots. The router's expert choice is untouched, so the output matches a non-streamed run.
void llama_moe_stream_wave_ids(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));
    GGML_ASSERT(dst->data != a->data); // the emit must not clobber the ids other waves read

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_wave_calls++;

    if (w == 0) {
        mgr->count_token_stats_locked(*sl, ids, (uint32_t) a->ne[0], a->ne[1]);
        mgr->plan_waves_locked(*sl, ids, n);
    }
    GGML_ASSERT(sl->plan_next_wave == w); // waves must run in order (enforced by the graph ordering token)

    const uint32_t n_ids = (uint32_t) a->ne[0]; // experts per token (n_expert_used)

    mgr->stage_wave_locked(lk, *sl, w, n_ids); // make this wave resident, preload the next, build the pool
    sl->plan_next_wave = w + 1;

    mgr->emit_wave_slots(*sl, ids, out, w, n_ids, a->ne[1]);
}

// multi-pass prefill: 1.0 for pairs whose expert belongs to wave w, 0.0 otherwise; multiplied into
// this wave's expert GEMM output so the masked-out (parked) pairs contribute nothing to the sum
void llama_moe_stream_wave_mask(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          float   * out = (float *) dst->data;

    std::lock_guard<std::mutex> lock(mgr->mtx);

    GGML_ASSERT(sl->plan_next_wave > w); // this wave's ids op has already run

    for (int64_t i = 0; i < n; i++) {
        out[i] = sl->expert_wave[ids[i]] == (uint8_t) w ? 1.0f : 0.0f;
    }
}
