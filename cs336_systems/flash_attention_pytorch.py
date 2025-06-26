import torch
from einops import rearrange

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Q, K, V: (batch_size, num_heads, seq_len, head_dim)
        is_causal: whether to apply causal mask
        """
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        B, N_q, D = Q.shape
        B, N_k, D = K.shape
        B, N_v, D = V.shape
        
        dtype_acc = torch.float32
        dtype_out = Q.dtype
        
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
       
        T_q = (N_q + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        T_k = (N_k + K_TILE_SIZE - 1) // K_TILE_SIZE

        O = torch.zeros_like(Q)
        L = torch.zeros((B, N_q), dtype=dtype_acc, device=Q.device)
        scale = 1.0 / D**0.5

        for i in range(T_q):
            q_start = i * Q_TILE_SIZE
            q_end = min(q_start + Q_TILE_SIZE, N_q)
            q_tile_size = q_end - q_start

            # O_i is the output of the i-th query tile (B, q_tile_size, D)
            O_i = torch.zeros((B, q_tile_size, D), dtype=dtype_acc, device=Q.device)
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

        return O
        

    
    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (batch_size, num_heads, seq_len, head_dim)
        """
        NotImplementedError("Backward pass not implemented")
        
        