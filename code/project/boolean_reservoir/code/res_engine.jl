using PythonCall
using LinearAlgebra

mutable struct ReservoirEngine
    nb_vals::Vector{Int64}              # CSR: flattened 1-indexed neighbor indices
    nb_offsets::Vector{Int64}           # CSR: nb_offsets[n]:nb_offsets[n+1]-1 are neighbors of n
    nb_shifts::Vector{Int64}            # pre-computed bit-shifts (k_i - pos) per neighbor
    clamped_nodes::Vector{Int64}        # 1-indexed nodes a tick leaves alone (input, output)
    active_nodes::Vector{Int64}         # 1-indexed nodes a tick computes: all but the clamped
    lut_flat::Vector{UInt8}             # jagged LUT: all nodes' entries concatenated
    lut_offsets::Vector{Int64}          # 1-indexed offsets: node n's LUT starts at lut_offsets[n]
    # The engine's own state: a double buffer (batch_size, N_total), Julia column-major,
    # so each node's samples are contiguous — a tick reads an input for all samples in
    # one sequential sweep. Loaded from Python's (row-major) states once per call and
    # stored back once at the end; the buffers swap each tick instead of copying.
    cur::Matrix{UInt8}
    nxt::Matrix{UInt8}
    idx::Vector{UInt32}                 # per-sample LUT index scratch (batch_size,)
    N_total::Int
    max_k::Int
    # Cached per-model constants (avoids pyconvert on every forward call)
    w_bi::Matrix{UInt8}                 # (c*k, N_I) — input connection weights
    ticks::Vector{UInt8}                # (c,) — reservoir ticks per chunk
    chunk_masks::Vector{Vector{Bool}}   # per-chunk input-node masks (precomputed from w_bi + pert)
    chunk_bits::Vector{Vector{Vector{Int}}}  # chunk_bits[ci][ni]: the chunk's bits wired to I-node ni (w_bi != 0)
    input_nodes::Vector{Int64}          # 1-indexed global positions of I-nodes (from selector)
    N_I::Int                            # number of input nodes (for chunk_masks/w_bi sizing)
    c::Int                              # number of input chunks
    k_bits::Int                         # bits per chunk
    pert::Int                           # perturbation mode: xor=0, override=1, and=2, or=3
    # Readout layer (weights pushed from Python after ridge solve; updated in-place for FORCE)
    output_indices::Vector{Int64}       # 1-indexed positions of output nodes in state vector
    readout_W::Matrix{Float32}          # (n_out, n_r) — readout weight matrix
    readout_b::Vector{Float32}          # (n_out,)    — readout bias
end

"""
Convert Python's flat jagged LUT + 0-based offsets to native Julia arrays.
lut_offsets is shifted to 1-based so hot-path access is `lut_flat[lut_offsets[n] + idx]`.
"""
function build_lut_jagged(lut_flat_py, lut_offsets_py)
    lut_flat    = pyconvert(Vector{UInt8}, lut_flat_py)
    lut_offsets = pyconvert(Vector{Int64}, lut_offsets_py) .+ Int64(1)  # 0-based → 1-based
    return lut_flat, lut_offsets
end

"""
Build CSR adjacency (nb_vals, nb_offsets, nb_shifts) from a padded adj_list + mask.
nb_shifts use per-node relative bit positions: shift = k_i - pos (MSB-first within
each node's actual degree k_i), so LUT index is always in [0, 2^k_i - 1].
"""
function build_csr_adj(adj_list, adj_mask, N_total::Int, max_k::Int)
    nb_vals    = Int64[]
    nb_offsets = Int64[1]
    nb_shifts  = Int64[]
    for n in 1:N_total
        k_i = sum(view(adj_mask, n, :))   # actual in-degree
        pos = 0
        for k in 1:max_k
            if adj_mask[n, k]
                pos += 1
                push!(nb_vals,   adj_list[n, k])
                push!(nb_shifts, k_i - pos)    # per-node MSB-first
            end
        end
        push!(nb_offsets, length(nb_vals) + 1)
    end
    return nb_vals, nb_offsets, nb_shifts
end

"""
    init_engine(adj_list_py, adj_list_mask_py, clamped_py,
                lut_flat_py, lut_offsets_py,
                w_bi_py, ticks_py, input_nodes_py, output_mask_py,
                N_total, max_k, batch_size, N_I, c, k_bits, pert, n_out)

Called once from Python at model init. Copies reference arrays into Julia memory
and pre-allocates the double-buffer. Readout weights initialised to zeros; call
set_readout_b after ridge solve to install trained weights.
"""
function init_engine(adj_list_py, adj_list_mask_py, clamped_py,
                     lut_flat_py, lut_offsets_py,
                     w_bi_py, ticks_py, input_nodes_py, output_mask_py,
                     N_total::Int, max_k::Int, batch_size::Int,
                     N_I::Int, c::Int, k_bits::Int, pert::Int, n_out::Int)
    adj_list      = pyconvert(Matrix{Int64}, adj_list_py) .+ Int64(1)
    adj_mask      = pyconvert(Matrix{Bool},  adj_list_mask_py)
    clamped       = pyconvert(Vector{Int64}, clamped_py) .+ Int64(1)
    input_nodes   = pyconvert(Vector{Int64}, input_nodes_py)   .+ Int64(1)
    output_mask   = pyconvert(Vector{Bool},  output_mask_py)

    lut_flat, lut_offsets          = build_lut_jagged(lut_flat_py, lut_offsets_py)
    nb_vals, nb_offsets, nb_shifts = build_csr_adj(adj_list, adj_mask, N_total, max_k)

    w_bi  = pyconvert(Matrix{UInt8}, w_bi_py)
    ticks = pyconvert(Vector{UInt8}, ticks_py)

    needs_mask = (pert == 1 || pert == 2)
    chunk_masks = Vector{Vector{Bool}}(undef, c)
    row = 1
    for ci in 1:c
        row_end = row + k_bits - 1
        if needs_mask
            mask = Vector{Bool}(undef, N_I)
            for n in 1:N_I
                connected = false
                for r in row:row_end
                    if w_bi[r, n] != 0x00
                        connected = true
                        break
                    end
                end
                mask[n] = connected
            end
            chunk_masks[ci] = mask
        else
            chunk_masks[ci] = trues(N_I)
        end
        row += k_bits
    end

    # w_bi is sparse (identity: one bit per I-node): list each node's wired bits once
    # instead of scanning every row of w_bi on every input step.
    chunk_bits = [[[bit for bit in 1:k_bits if w_bi[(ci - 1) * k_bits + bit, ni] != 0x00] for ni in 1:N_I]
                  for ci in 1:c]

    cur            = zeros(UInt8, batch_size, N_total)
    nxt            = zeros(UInt8, batch_size, N_total)
    idx            = zeros(UInt32, batch_size)
    output_indices = findall(output_mask)   # 1-indexed
    n_r            = length(output_indices)
    readout_W      = zeros(Float32, n_out, n_r)
    readout_b      = zeros(Float32, n_out)

    active = setdiff(1:N_total, clamped)
    return ReservoirEngine(nb_vals, nb_offsets, nb_shifts, clamped, active, lut_flat, lut_offsets,
                           cur, nxt, idx, N_total, max_k, w_bi, ticks, chunk_masks, chunk_bits,
                           input_nodes, N_I, c, k_bits, pert,
                           output_indices, readout_W, readout_b)
end

"""
    set_readout_b(engine, W_py, b_py)

Push trained readout weights from Python into the Julia engine.
Call after ridge solve. Also the update point for future FORCE learning.
Python access: jl.set_readout_b(engine, W, b)
"""
function set_readout!(engine::ReservoirEngine, W_py, b_py)
    engine.readout_W = pyconvert(Matrix{Float32}, W_py)
    engine.readout_b = pyconvert(Vector{Float32}, b_py)
    return nothing
end

"""
    bipolar_shift!(o, bipolar)

Single home for the bipolar state shift {0,1} → {−1,+1} (O.encoding='bipolar').
Shared by both readout modes; the Python twin is `_reduce_states._shift`.
"""
function bipolar_shift!(o::AbstractMatrix{Float32}, bipolar::Bool)
    if bipolar
        @. o = o * 2.0f0 - 1.0f0
    end
    return o
end

"""
    apply_readout_b(engine, states_py, m, bipolar)

Extract output nodes from states, apply float conversion, bipolar shift if needed,
then compute W @ o + b. Returns (m, n_out) Float32 matrix.
For classification mode (final state).
Python access: jl.apply_readout_b(engine, states, m, bipolar)
"""
function apply_readout!(engine::ReservoirEngine, states_py::AbstractMatrix{UInt8},
                          m::Int, bipolar::Bool)
    n_r   = length(engine.output_indices)
    n_out = length(engine.readout_b)
    o = Matrix{Float32}(undef, m, n_r)
    @inbounds for (ri, ni) in enumerate(engine.output_indices)
        @simd for s in 1:m
            o[s, ri] = Float32(states_py[s, ni])
        end
    end
    bipolar_shift!(o, bipolar)
    # (m, n_r) * (n_r, n_out) + (n_out,) broadcast → (m, n_out)
    return o * engine.readout_W' .+ engine.readout_b'
end

"""
    apply_readout_timeseries_b(engine, step_buffer_py, m, s, bipolar)

Time-series readout: apply W @ o_step + b for each step, sum over steps.
step_buffer_py shape: (s, m, N_total). Returns (m, n_out) Float32.
Python access: jl.apply_readout_timeseries_b(engine, step_buffer, m, s, bipolar)
"""
function apply_readout_timeseries!(engine::ReservoirEngine, step_buffer_py, m::Int, s::Int, bipolar::Bool)
    step_buffer = pyconvert(Array{UInt8, 3}, step_buffer_py)  # (s, m, N_total)
    n_r   = length(engine.output_indices)
    n_out = length(engine.readout_b)
    outputs = zeros(Float32, m, n_out)
    o = Matrix{Float32}(undef, m, n_r)
    for si in 1:s
        @inbounds for (ri, ni) in enumerate(engine.output_indices)
            @simd for samp in 1:m
                o[samp, ri] = Float32(step_buffer[si, samp, ni])
            end
        end
        bipolar_shift!(o, bipolar)
        outputs += o * engine.readout_W' .+ engine.readout_b'
    end
    return outputs
end

"""
    load_states!(engine, states_py, m) / store_states!(engine, states_py, m)

Copy rows 1..m between Python's states (row-major numpy) and the engine's buffer.
"""
function load_states!(engine::ReservoirEngine, states_py::AbstractMatrix{UInt8}, m::Int)
    cur = engine.cur
    @inbounds for s in 1:m, n in 1:engine.N_total
        cur[s, n] = states_py[s, n]
    end
end

function store_states!(engine::ReservoirEngine, states_py::AbstractMatrix{UInt8}, m::Int)
    cur = engine.cur
    @inbounds for s in 1:m, n in 1:engine.N_total
        states_py[s, n] = cur[s, n]
    end
end

"""
    tick!(engine, m)

Synchronous Boolean update of the engine's buffer, m samples in parallel. Node n's next
state is lut[offset_n + idx], idx = its inputs' states MSB-first: built one input at a
time across all samples (contiguous, SIMD), then looked up. Clamped nodes are not
computed, they keep their state. The buffers then swap.
"""
function tick!(engine::ReservoirEngine, m::Int)
    cur, nxt, idx = engine.cur, engine.nxt, engine.idx
    nb_vals, nb_shifts, nb_offsets = engine.nb_vals, engine.nb_shifts, engine.nb_offsets
    lut, lut_offsets = engine.lut_flat, engine.lut_offsets

    @inbounds for n in engine.active_nodes
        lo = nb_offsets[n]
        hi = nb_offsets[n + 1] - 1
        base = lut_offsets[n]
        if lo > hi                       # k = 0: the 0-ary table, a constant
            v = lut[base]
            @simd for s in 1:m
                nxt[s, n] = v
            end
            continue
        end
        nb, sh = nb_vals[lo], nb_shifts[lo]
        @simd for s in 1:m
            idx[s] = UInt32(cur[s, nb]) << sh
        end
        for ki in lo+1:hi
            nb, sh = nb_vals[ki], nb_shifts[ki]
            @simd for s in 1:m
                idx[s] |= UInt32(cur[s, nb]) << sh
            end
        end
        for s in 1:m
            nxt[s, n] = lut[base + idx[s]]
        end
    end

    @inbounds for n in engine.clamped_nodes
        @simd for s in 1:m
            nxt[s, n] = cur[s, n]
        end
    end

    engine.cur, engine.nxt = nxt, cur
    return nothing
end

"""
    reservoir_tick!(engine, states_py, m)

One synchronous tick of Python's states (load, tick!, store).
"""
function reservoir_tick!(engine::ReservoirEngine, states_py::AbstractMatrix{UInt8}, m::Int)
    load_states!(engine, states_py, m)
    tick!(engine, m)
    store_states!(engine, states_py, m)
end

"""
    warmup_b(engine, states_py, m, ticks)

Run `ticks` reservoir ticks without input to wash out the init state.
Python access: jl.warmup_b(engine, states_py, m, ticks)
"""
function warmup!(engine::ReservoirEngine, states_py::AbstractMatrix{UInt8}, m::Int, ticks::Int)
    load_states!(engine, states_py, m)
    for _ in 1:ticks
        tick!(engine, m)
    end
    store_states!(engine, states_py, m)
end

"""
    forward_sequence_b(engine, states_py, x_py, history_buffer, step_buffer)

Full Boolean forward pass (s × c × t loop) in Julia.
- states_py     : AbstractMatrix{UInt8} (batch_size, N_total), modified in-place
- x_py          : Fortran-order Array{UInt8,4} (m, s, c, k)
- history_buffer: PyArray{UInt8,3} (n_entries, m, N_total) or Python None
- step_buffer   : PyArray{UInt8,3} (s, m, N_total) or Python None (for time-series readout)

Python access: jl.forward_sequence_b(engine, states, x, history_buffer, step_buffer)
"""
function forward_sequence!(engine::ReservoirEngine, states_py::AbstractMatrix{UInt8},
                              x_py, history_buffer, step_buffer)
    x = pyconvert(Array{UInt8, 4}, x_py)
    m = size(x, 1)
    s = size(x, 2)

    has_history  = !pyis(history_buffer, pybuiltins.None)
    has_stepbuf  = !pyis(step_buffer,   pybuiltins.None)

    pert_bits = zeros(UInt8, m)
    pert = engine.pert       # xor=0, override=1, and=2, or=3
    entry_idx = 1
    load_states!(engine, states_py, m)

    for si in 1:s
        for ci in 1:engine.c
            has_conn = engine.chunk_masks[ci]
            wired    = engine.chunk_bits[ci]

            @inbounds for ni in eachindex(engine.input_nodes)
                n = engine.input_nodes[ni]
                has_conn[ni] || continue

                fill!(pert_bits, 0x00)
                for bit in wired[ni]
                    @simd for samp in 1:m
                        pert_bits[samp] |= x[samp, si, ci, bit]
                    end
                end

                cur = engine.cur
                if pert == 0
                    @simd for samp in 1:m; cur[samp, n] ⊻= pert_bits[samp]; end
                elseif pert == 1
                    @simd for samp in 1:m; cur[samp, n] = pert_bits[samp]; end
                elseif pert == 2
                    @simd for samp in 1:m; cur[samp, n] &= pert_bits[samp]; end
                else
                    @simd for samp in 1:m; cur[samp, n] |= pert_bits[samp]; end
                end
            end

            if has_history
                @inbounds for n in 1:engine.N_total
                    @simd for samp in 1:m
                        history_buffer[entry_idx, samp, n] = engine.cur[samp, n]
                    end
                end
                entry_idx += 1
            end

            t = Int(engine.ticks[ci])
            for ti in 1:t
                tick!(engine, m)

                is_last = (si == s) && (ci == engine.c) && (ti == t)
                if has_history && !is_last
                    @inbounds for n in 1:engine.N_total
                        @simd for samp in 1:m
                            history_buffer[entry_idx, samp, n] = engine.cur[samp, n]
                        end
                    end
                    entry_idx += 1
                end
            end
        end

        if has_stepbuf
            @inbounds for n in 1:engine.N_total
                @simd for samp in 1:m
                    step_buffer[si, samp, n] = engine.cur[samp, n]
                end
            end
        end
    end

    if has_history
        @inbounds for n in 1:engine.N_total
            @simd for samp in 1:m
                history_buffer[entry_idx, samp, n] = engine.cur[samp, n]
            end
        end
    end

    store_states!(engine, states_py, m)
    return states_py
end
