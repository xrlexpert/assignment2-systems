import torch
from einops import rearrange,einsum,reduce

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Q, K, V: (batch_size, num_heads, seq_len, head_dim)
        is_causal: whether to apply causal mask
        """
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        batch_dims, N_q, Dim = Q.shape[:-2], Q.shape[-2], Q.shape[-1]
        N_k = K.shape[-2]
        N_v = V.shape[-2]
        assert N_k == N_v, "K and V must have the same number of keys"

        Q = rearrange(Q, "... n_q d -> (...) n_q d")
        K = rearrange(K, "... n_k d -> (...) n_k d")
        V = rearrange(V, "... n_v d -> (...) n_v d")
        B = Q.shape[0]
        dtype_acc = torch.float32
        dtype_out = Q.dtype
        
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
       
        T_q = (N_q + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        T_k = (N_k + K_TILE_SIZE - 1) // K_TILE_SIZE

        O = torch.zeros_like(Q, device=Q.device, dtype=torch.float32)
        L = torch.zeros((B, N_q), dtype=dtype_acc, device=Q.device)
        scale = 1.0 / Dim**0.5

        for i in range(T_q):
            q_start = i * Q_TILE_SIZE
            q_end = min(q_start + Q_TILE_SIZE, N_q)
            q_tile_size = q_end - q_start

            # O_i is the output of the i-th query tile (B, q_tile_size, D)
            O_i = torch.zeros((B, q_tile_size, Dim), dtype=dtype_acc, device=Q.device)
            # l_i is the row-wise sum of S_ij (B, q_tile_size)
            l_i = torch.zeros((B, q_tile_size), dtype=dtype_acc, device=Q.device)
            # m_i is the row-wise max of S_ij (B, q_tile_size)
            m_i = torch.full((B, q_tile_size), -torch.inf, dtype=dtype_acc, device=Q.device)

            # iterate over key tiles
            for j in range(T_k):
                k_start = j * K_TILE_SIZE
                k_end = min(k_start + K_TILE_SIZE, N_k)
                k_tile_size = k_end - k_start

                Q_tile = Q[:, q_start:q_end, :].to(dtype_acc)
                K_tile = K[:, k_start:k_end, :].to(dtype_acc)
                V_tile = V[:, k_start:k_end, :].to(dtype_acc)
               
                # (B, q_tile_size, k_tile_size)
                s_ij = Q_tile @ K_tile.transpose(-2, -1) * scale
                if is_causal:
                    mask = torch.triu(torch.ones(q_tile_size, k_tile_size, dtype=torch.bool, device=Q.device), diagonal=1)
                    s_ij.masked_fill_(mask, -torch.inf)
                
                m_iprev = m_i
                m_i = torch.maximum(m_iprev, torch.max(s_ij, dim=-1)[0])
                
                # We use .detach() to prevent gradients from flowing back through m_i and m_iprev.
                p_ij = torch.exp(s_ij - m_i.detach().unsqueeze(-1))

                l_scale_factor = torch.exp(m_iprev.detach() - m_i.detach())

                l_i = l_scale_factor * l_i + torch.sum(p_ij, dim=-1)

                O_i =  rearrange(l_scale_factor, 'b q -> b q 1') * O_i + (p_ij @ V_tile)
            
            l_i_safe = l_i + 1e-6

            O_i_final = O_i * rearrange(1.0 / l_i_safe, 'b q -> b q 1')

            logsumexp_final = m_i + torch.log(l_i_safe)

            O[:, q_start:q_end, :] = O_i_final.to(dtype_out)
            L[:, q_start:q_end] = logsumexp_final
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.batch_dims = batch_dims

        O = O.reshape(*batch_dims, N_q, Dim)
        return O
        
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        batch_dims = ctx.batch_dims

        # reshape dO
        dO = dO.reshape(-1, dO.shape[-2], dO.shape[-1])

        B, N_q, Dim = Q.shape
        _, N_k, _ = K.shape
        _, N_v, _ = V.shape
        device = Q.device
        dtype = Q.dtype
        
        scale = 1/(Dim**0.5)
        D = O * dO # element-wise product, b x N_q x D_v
        D = reduce(D, '... q d -> ... q', 'sum') # rowsum of D
        S = einsum(Q, K, '... q d, ... k d -> ... q k') * scale

        if ctx.is_causal:
            # just do a torch mask for the backward pass
            mask = torch.tril(torch.ones(N_q, N_k, device = device, dtype = dtype))
            S = S.masked_fill(mask == 0, -float('inf'))

        P_ij = torch.exp(S - L.unsqueeze(-1))
        dV = einsum(P_ij, dO, '... q k, ... q d -> ... k d') # N_k = N_v
        dP = einsum(dO, V, '... q d, ... v d -> ... q v')
        dS_ij = P_ij * (dP - D.unsqueeze(-1))
        dQ = einsum(dS_ij, K, '... q k, ... k d -> ... q d') * scale
        dK = einsum(dS_ij, Q, '... q k, ... q d -> ... k d') * scale
        
        dQ = dQ.reshape(*batch_dims, N_q, Dim)
        dK = dK.reshape(*batch_dims, N_k, Dim)
        dV = dV.reshape(*batch_dims, N_k, Dim)
        # return None corresponding to "causal"
        return dQ, dK, dV, None
    
    # @staticmethod
    # def backward(ctx, dO):
    #     """
    #     dO: (..., q_len, d_model)
    #     """
    #     Q, K, V, O, L = ctx.saved_tensors
    #     scale = ctx.scale
    #     is_causal = ctx.is_causal
    #     batch_dims = ctx.batch_dims

    #     B, N_q, Dim = Q.shape
    #     _, N_k, _ = K.shape

    #     dtype_acc = torch.float32
    #     dtype_out = Q.dtype

    #     Q_TILE_SIZE = 64
    #     K_TILE_SIZE = 64

    #     T_q = (N_q + Q_TILE_SIZE - 1) // Q_TILE_SIZE
    #     T_k = (N_k + K_TILE_SIZE - 1) // K_TILE_SIZE

    #     dO = dO.reshape(-1, N_q, Dim)

    #     dQ = torch.zeros_like(Q, device=Q.device, dtype=torch.float32)
    #     dK = torch.zeros_like(K, device=K.device, dtype=torch.float32)
    #     dV = torch.zeros_like(V, device=V.device, dtype=torch.float32)
    #     D = torch.sum(dO * O, dim=-1) # （B, N_q,)

    #     # Traversal strategy comparison:
    #     # Strategy 1: Outer loop over Q tiles, inner loop over K and V tiles
    #     # - For each Q_i, we iterate over all K_j and V_j
    #     # - Each K_j and V_j is repeatedly loaded for every Q_i  
    #     # - Inner loop writes to dK_j and dV_j
    #     # - Total memory access: high, due to frequent reloading of K and V

    #     # Strategy 2: Outer loop over K and V tiles, inner loop over Q tiles
    #     # - For each K_j and V_j, we iterate over all Q_i
    #     # - Each Q_i is reused within the inner loop 
    #     # - Inner loop writes to dQ_i, 
    #     # - Total memory access: lower, innor loop only write to dQ (one) instead of dK and dV(two)
    #     for j in range(T_k):
    #         k_start = j * K_TILE_SIZE
    #         k_end = min(k_start + K_TILE_SIZE, N_k)
    #         k_tile_size = k_end - k_start

    #         K_tile = K[:, k_start:k_end, :].to(dtype_acc)
    #         V_tile = V[:, k_start:k_end, :].to(dtype_acc)

    #         dK_j = torch.zeros((B, k_tile_size, Dim), dtype=dtype_acc, device=Q.device)
    #         dV_j = torch.zeros((B, k_tile_size, Dim), dtype=dtype_acc, device=Q.device)

    #         for i in range(T_q):
    #             q_start = i * Q_TILE_SIZE
    #             q_end = min(q_start + Q_TILE_SIZE, N_q)
    #             q_tile_size = q_end - q_start

    #             Q_tile = Q[:, q_start:q_end, :].to(dtype_acc)
    #             dO_i = dO[:, q_start:q_end, :].to(dtype_acc)
    #             D_i = D[:, q_start:q_end].to(dtype_acc)  # (B, q_tile_size)
    #             L_i = L[:, q_start:q_end].to(dtype_acc)

    #             S_ij = Q_tile @ K_tile.transpose(-2, -1) * scale
    #             if is_causal:
    #                 mask = torch.triu(
    #                     torch.ones(q_tile_size, k_tile_size, dtype=torch.bool, device=Q.device),
    #                     diagonal=1
    #                 )
    #                 S_ij.masked_fill_(mask, -torch.inf)

    #             P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))  # (B, q, k)
    #             dP_ij = dO_i @ V_tile.transpose(-2, -1)        # (B, q, k)
    #             dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1)) * scale

    #             dQ[:, q_start:q_end, :] += (dS_ij @ K_tile).to(dtype_out)
    #             dK_j += dS_ij.transpose(-2, -1) @ Q_tile
    #             dV_j += P_ij.transpose(-2, -1) @ dO_i

    #         dK[:, k_start:k_end, :] += dK_j.to(dtype_out)
    #         dV[:, k_start:k_end, :] += dV_j.to(dtype_out)

    #     dQ = dQ.reshape(*batch_dims, N_q, Dim)
    #     dK = dK.reshape(*batch_dims, N_k, Dim)
    #     dV = dV.reshape(*batch_dims, N_k, Dim)
    #     return dQ, dK, dV, None