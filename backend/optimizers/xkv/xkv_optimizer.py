import torch
from kv_cache.utils import extract_keys_and_values
import numpy as np

# TODO : test lại hàm tính CKA theo cách tác giả, hiện tại đang dùng hàm tính CKA chuẩn (linear CKA) để tính tỉ lệ đồng dạng giữa các layer liên tiếp
# đây là hàm tinh CKA theo cách tác giả, tính riêng cho từng head và batch --> không nén token --> tính tỉ lệ đồng dạng cka giưa các layer liên tiếp
def compute_author_cka(X, Y, eps=1e-12):
    """
    Tính Linear CKA theo cách tác giả.

    Input:
        X, Y: [B, H, L, D]

    Output:
        cka_per_head: [B, H]
        cka_mean: scalar

    Mỗi CKA được tính riêng trên L token
    của từng batch và từng attention head.
    """

    B, H, L, D = X.shape

    assert X.shape == Y.shape, "X và Y phải cùng shape"

    X = X.to(torch.float32)
    Y = Y.to(torch.float32)

    cka_per_head = torch.zeros(
        B, H,
        device=X.device,
        dtype=torch.float32
    )

    for b in range(B):
        for h in range(H):

            # Lấy biểu diễn token của từng head
            x = X[b, h]  # [L, D]
            y = Y[b, h]  # [L, D]

            # Tạo Gram matrix: [L, L]
            K = x @ x.T
            L_gram = y @ y.T

            # Centering Gram matrix
            n = K.shape[0]
            I = torch.eye(n, device=X.device)
            ones = torch.ones(
                n, n,
                device=X.device
            )

            H_center = I - ones / n

            K_centered = H_center @ K @ H_center
            L_centered = H_center @ L_gram @ H_center

            # Linear CKA
            numerator = torch.sum(
                K_centered * L_centered
            )

            denominator = (
                torch.norm(K_centered, p="fro")
                * torch.norm(L_centered, p="fro")
            )

            cka_per_head[b, h] = (
                numerator / denominator.clamp_min(eps)
            )

    # Trung bình các batch và head
    cka_mean = cka_per_head.mean()

    return cka_per_head, cka_mean.item()


def compute_liner_cka(X,Y):
    """
         tính linear CKA giữa 2 tensor [seq_len,hidden_dim]
         cần flatten các chiều batch và head về 2D trước khi tính CKA    
    """
    # flatten các chiều batch và head về 2D
    X_flat = X.reshape(-1, X.shape[-1]).to(torch.float32)
    Y_flat = Y.reshape(-1, Y.shape[-1]).to(torch.float32)

    # center dữ liệu (trừ đi mean)
    X_centered = X_flat - X_flat.mean(dim=0)
    Y_centered = Y_flat - Y_flat.mean(dim=0)

    # tính CKA
    dot_prod = torch.norm(torch.matmul(X_centered.T, Y_centered), p='fro') ** 2
    norm_X = torch.norm(torch.matmul(X_centered.T, X_centered), p='fro')
    norm_Y = torch.norm(torch.matmul(Y_centered.T, Y_centered), p='fro')

    if norm_X == 0 or norm_Y == 0:
        return 0.0
    return (dot_prod / (norm_X * norm_Y)).item()

def to_2d(tensor):
    """
        Chuyển tensor 4D [B, H, S, D] thành 2D [B*H*S, D]
    """
    assert tensor.shape[0] == 1 , "bản demo chỉ hỗ trợ batch_size = 1"
    _ , H , S , D = tensor.shape
    return tensor[0].permute(1,0,2).reshape(S, H * D)

def from_2d(m,H,D):
    """[S,H*D] --> [1 ,H ,S ,D]"""
    return m.reshape(m.shape[0],H,D).permute(1,0,2).unsqueeze(0)

class XKVOptimizer:
    def __init__(self, rank_ratio_k =0.09 , rank_ratio_v = 0.14 ,window_size = 4 , grouping = "window", cka_threshold = 0.85 ):
        """
        rank_ratio_*: r / (G*d). Khi S lớn, tỉ lệ nén ≈ 1/rank_ratio.
                      (bài báo: W=4, d=1024, r_k=384, r_v=576 -> 0.094 và 0.14)
        grouping: "window" (layer liền kề, đúng bài báo) hoặc "cka" (greedy của bạn)
        """
        assert grouping in ("window", "cka")
        self.rank_ratio_k = rank_ratio_k
        self.rank_ratio_v = rank_ratio_v
        self.W = window_size
        self.grouping = grouping
        self.cka_threshold = cka_threshold

    def _make_groups(self,keys,values):
        n = len(keys)
        if self.grouping == "window":
            return [list(range(s,min(s+self.W,n))) for s in range(0,n,self.W)]

        k2d = [to_2d(k) for k in keys]
        v2d = [to_2d(v) for v in values]

        assigned , groups = [False] * n , []
        for i in range(n):
            if assigned[i] : continue
            assigned[i] = True
            g = [i]
            for j in range(i+1,n):
                if assigned[j] : continue
                if compute_liner_cka(k2d[i],k2d[j]) >= self.cka_threshold and compute_liner_cka(v2d[i],v2d[j]) >= self.cka_threshold:
                    g.append(j)
                    assigned[j] = True
            groups.append(g)
        return groups 

    # ---------- phân rã chéo layer ----------
    @staticmethod
    def _factorize(mats , rank_raito):
        """mats: list G ma trận [S,d] -> A[S,r] chung Bs: List G ma trận [r,d]"""
        d = mats[0].shape[1]
        big = torch.cat([m.float() for m in mats] , dim = 1)
        r = min(max(1, int(big.shape[1] * rank_raito)) , *big.shape)
        U, S , Vh = torch.linalg.svd(big, full_matrices = False)
        A = (U[:,:r] * S [ : r]).half() # [S, r]   dùng chung
        Vr = Vh [:r] # [r, G*d]
        Bs = [Vr[:, i * d:(i + 1) * d].half() for i in range(len(mats))]  # [r, d] riêng từng layer
        return A , Bs

    def compress(self, past_key_values):
        keys, values = extract_keys_and_values(past_key_values)
        _, self.H, _, self.D = keys[0].shape
        self.num_layers = len(keys)
        self.layer_to_group = {}
        self.groups = []
        for gi, ids in enumerate(self._make_groups(keys, values)):
            Ak, Bk = self._factorize([to_2d(keys[i]) for i in ids], self.rank_ratio_k)
            Av, Bv = self._factorize([to_2d(values[i]) for i in ids], self.rank_ratio_v)
            self.groups.append({"ids": ids, "Ak": Ak, "Bk": Bk, "Av": Av, "Bv": Bv})
            for pos, lid in enumerate(ids):
                self.layer_to_group[lid] = (gi, pos)
        return self.groups

    compress_with_cka = compress   # giữ tên cũ cho main.py

    # ---------- khôi phục đúng 1 layer ----------
    def reconstruct_layer(self, layer_idx):
        gi, pos = self.layer_to_group[layer_idx]
        g = self.groups[gi]
        k = g["Ak"].float() @ g["Bk"][pos].float()    # [S, r] @ [r, d] -> [S, d]
        v = g["Av"].float() @ g["Bv"][pos].float()
        return from_2d(k, self.H, self.D), from_2d(v, self.H, self.D)

    def memory_mb(self):
        total = 0
        for g in self.groups:
            for key in ("Ak", "Av"):
                total += g[key].nelement() * g[key].element_size()
            for key in ("Bk", "Bv"):
                total += sum(b.nelement() * b.element_size() for b in g[key])
        return total / 1024 ** 2