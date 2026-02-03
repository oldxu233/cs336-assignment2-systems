import torch


class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, is_causal=False):
        """
        query: (B, Q, D)
        key:   (B, K, D)
        value: (B, K, D)
        """
        batch_size, seq_len, d = query.shape
        K = key.shape[1]
        softmax_scale = 1 / (d ** 0.5)

        O = torch.zeros((batch_size, seq_len, d), device=query.device)
        L = torch.empty((batch_size, seq_len), device=query.device, dtype=query.dtype)
        
        Bq, Bk = 16, 16

        for qs in range(0, seq_len, Bq):
            qe = min(qs + Bq, seq_len)
            q = query[:, qs:qe, :]
            
            o_i = torch.zeros((batch_size, qe - qs, value.shape[-1]), device=query.device)
            m_i = torch.full((batch_size, qe - qs), -float("inf"), device=query.device)
            l_i = torch.zeros((batch_size, qe - qs), device=query.device)

            for ks in range(0, K, Bk):
                ke = min(ks + Bk, K)
                k = key[:, ks:ke, :]
                v = value[:, ks:ke, :]
                scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
                
                if is_causal:
                    q_idx = torch.arange(qs, qe, device=query.device)[:, None]
                    k_idx = torch.arange(ks, ke, device=query.device)[None, :]
                    scores = scores.masked_fill(q_idx < k_idx, -float("inf"))
                
                m_new = torch.maximum(m_i, scores.max(dim=-1).values)
                p = torch.exp(scores - m_new.unsqueeze(-1))
                
                l_new = torch.exp(m_i - m_new) * l_i + p.sum(dim=-1)
                o_new = torch.exp(m_i - m_new).unsqueeze(-1) * o_i + torch.matmul(p, v)
                
                m_i = m_new
                l_i = l_new 
                o_i = o_new

            O[:, qs:qe, :] = o_i / l_i.unsqueeze(-1)
            L[:, qs:qe] = m_i + torch.log(l_i)
        ctx.save_for_backward(L)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        pass