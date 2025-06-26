import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    is_causal: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )


    # Load the query tile
    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")


    # Accumulators
    o_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_acc = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_acc = tl.full((Q_TILE_SIZE,), -torch.inf, dtype=tl.float32)

    # Iterate over key tiles
    k_num_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(k_num_tiles):
        k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        q_tile_fp32 = q_tile.to(tl.float32)
        k_tile_fp32 = k_tile.to(tl.float32)
        

        s_ij = tl.dot(q_tile_fp32, k_tile_fp32.T) * scale
        if is_causal:
            q_rows = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_cols = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = (q_rows[:, None] >= k_cols[None, :])
            s_ij = tl.where(causal_mask, s_ij, -1e9)


        m_prev = m_acc
        m_acc = tl.maximum(m_prev, tl.max(s_ij, axis=-1))
        
        # triton need to implement backward exci
        p_ij = tl.exp(s_ij - m_acc[:, None])

        l_scale_factor = tl.exp(m_prev - m_acc)
        l_acc = l_scale_factor * l_acc + tl.sum(p_ij, axis=-1)

        o_acc = l_scale_factor[:, None] * o_acc + tl.dot(p_ij.to(v_tile.dtype), v_tile)

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    l_acc = tl.where(l_acc == 0.0, 1.0, l_acc)
    l_inverse = 1.0 / l_acc
    o_acc = o_acc * l_inverse[:, None].to(O_ptr.type.element_ty)
    logsumexp = m_acc + tl.log(l_acc)

    tl.store(O_block_ptr, o_acc, boundary_check=(0, 1))
    tl.store(L_block_ptr, logsumexp, boundary_check=(0,))

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, q_tile_size=64, k_tile_size=64):
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        B, N_q, D = Q.shape
        B, N_k, D = K.shape
        B, N_v, D = V.shape

        scale = 1.0/ (D**0.5)

        O = torch.zeros_like(Q)
        L = torch.zeros((B, N_q,), device=Q.device, dtype=torch.float32)

        grid = ((N_q + q_tile_size - 1) // q_tile_size, B)

        flash_fwd_kernel[grid](
            Q,K,V,O,L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            scale=scale,
            D=D,
            is_causal=is_causal,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = scale

        return O



    