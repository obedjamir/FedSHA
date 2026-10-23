"""
Models (segmentation only). Every model is a frozen MedSAM ViT-B with the MedSAM mask decoder (no prompts) and
SAM's hypernetwork mask head; they differ in how the encoder is adapted.

  MSAModel   : one dense MSA adapter next to the attention and MLP branch of every block (MSA-based baselines).
  MENUModel  : Fed-MENU (Xu et al., 2023): one organ-specific encoder per organ (frozen ViT + that organ's adapters),
               features fused before the decoder, plus an auxiliary generic decoder (AGD).
  MoSAModel  : FedMoSA: a modality-specialised adapter (hard-routed by the batch's modality) and a shared adapter
               next to the attention and MLP branch of every block.
  FedSHA     : ours. One G x G grid of shared adapters used by all blocks and both branches. A depth-dependent
               sliding window moves from the centre of the grid (shallow blocks) to the client's organ region
               (deep blocks); inside the window a gate keeps the top-k adapters. Scale / shift per block and branch.

Parameter groups (param_group -> aggregation):
  frozen  : MedSAM encoder, prompt encoder (except the no-mask embedding), IoU head
  pool    : FedSHA adapter grid (one index per adapter)
  gate    : FedSHA gates (row e scores adapter e)
  film    : FedSHA scale / shift of the adapter branch
  shared  : MSA adapters, FedMoSA shared adapters, Fed-MENU fusion / AGD
  organ   : Fed-MENU organ-specific adapters
  spec    : FedMoSA modality-specialised adapters
  local   : FedDBE representation bias (never aggregated)
  decoder : decoder trunk (transformer, IoU token, no-mask embedding, upscaling) and, with --shared_head 1,
            the output layer
  slot / tokens : output layer (hypernetwork MLP j / mask token j) when it is kept local (--shared_head 0)

Parameter names (anatomy_pool, gate_attn, mod_gamma_attn, ...) are kept from earlier versions, so all saved
checkpoints load unchanged.
"""
import re
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from segment_anything import sam_model_registry
from segment_anything.modeling.image_encoder import window_partition, window_unpartition


# -------------------------------------------------------------------
# Bottleneck adapter
# -------------------------------------------------------------------
class Adapter(nn.Module):
    def __init__(self, hidden_size, reduction_factor=16, zero_init=True):
        super().__init__()
        self.down = nn.Linear(hidden_size, hidden_size // reduction_factor)
        self.relu = nn.ReLU()
        self.up = nn.Linear(hidden_size // reduction_factor, hidden_size)
        if zero_init:                               # start exactly at MedSAM
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.relu(self.down(x)))


def adapt_sam_resolution(sam, img_size=512):
    """Run SAM at img_size instead of 1024: absolute position embeddings interpolated to (img_size/16)^2;
    relative position tables are interpolated by SAM itself; masks come out at img_size/4."""
    enc, pe = sam.image_encoder, sam.prompt_encoder
    if enc.img_size == img_size:
        return sam
    grid = img_size // enc.patch_embed.proj.kernel_size[0]
    if enc.pos_embed is not None and enc.pos_embed.shape[1] != grid:
        pos = enc.pos_embed.data.permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(grid, grid), mode="bicubic", align_corners=False)
        enc.pos_embed = nn.Parameter(pos.permute(0, 2, 3, 1).contiguous(), requires_grad=False)
    enc.img_size = img_size
    pe.image_embedding_size = (grid, grid)
    pe.input_image_size = (img_size, img_size)
    return sam


# -------------------------------------------------------------------
# Parameter groups (aggregation)
# -------------------------------------------------------------------
_POOL_RE = re.compile(r"^anatomy_pool\.experts\.(\d+)\.")
_SLOT_RE = re.compile(r"^sam\.mask_decoder\.output_hypernetworks_mlps\.(\d+)\.")
DECODER_TRUNK = ("sam.mask_decoder.transformer", "sam.mask_decoder.iou_token",
                 "sam.mask_decoder.output_upscaling", "sam.prompt_encoder.no_mask_embed")
MASK_TOKENS = "sam.mask_decoder.mask_tokens.weight"
_ORGAN_RE = re.compile(r"\.menu_adapters\.(\d+)\.")
_SPEC_RE = re.compile(r"\.mosa_spec_(?:attn|mlp)\.([A-Za-z0-9]+)\.")
SHARED_HEAD = True     # every client predicts ONE class with the same output layer (mask token 0 + hypernetwork MLP 0),
                       # so the output layer is part of the decoder and aggregated with it (set by main.py --shared_head)


def is_head(name):
    """Output layer of the mask decoder (mask tokens + output hypernetwork MLPs)."""
    return bool(_SLOT_RE.match(name)) or name == MASK_TOKENS


def param_group(name):
    """-> (group, sub-key, index); see the module docstring for the groups."""
    parts = name.split(".")
    if name.startswith("dbe_"):
        return "local", None, None                       # FedDBE: personalized representation bias (never sent)
    if any(p.startswith("gate") for p in parts):
        return "gate", None, None                        # gates of the shared grid (row e = adapter e)
    if "mod_gamma" in name or "mod_beta" in name:
        return "film", None, None                        # scale / shift of the adapter branch
    m = _POOL_RE.match(name)
    if m:
        return "pool", None, int(m.group(1))
    if ".msa_adapter_" in name or ".mosa_shared_" in name or name.startswith(("menu_fuse.", "menu_agd.")):
        return "shared", None, None
    m = _ORGAN_RE.search(name)
    if m:
        return "organ", None, int(m.group(1))
    m = _SPEC_RE.search(name)
    if m:
        return "spec", m.group(1), None
    if name.startswith(DECODER_TRUNK):
        return "decoder", None, None
    m = _SLOT_RE.match(name)
    if m:
        if SHARED_HEAD:
            return "decoder", None, None                  # one output layer for all clients (aggregated as decoder)
        return "slot", None, int(m.group(1))
    if name == MASK_TOKENS:
        if SHARED_HEAD:
            return "decoder", None, None
        return "tokens", None, None
    return "frozen", None, None


# -------------------------------------------------------------------
# Sliding window on the adapter grid
# -------------------------------------------------------------------
def organ_key(name):
    """Organ name as sent to the server."""
    return name.strip().lower()


def modality_key(name):
    """Modality name as sent to the server."""
    return name.strip().lower()


class SlidingWindow:
    """Depth-dependent sliding window on the G x G adapter grid.
    The grid is cut into 2 x 2 organ regions (organ i owns region i, row-major); the centre region is the central
    2 x 2 of the grid (one adapter of each surrounding region). In block l (1..L) the 2 x 2 window is centred at
        mu_l = (1 - tau_l) mu_0 + tau_l mu_o,   tau_l = (l - 1) / (L - 1),
    from the grid centre (block 1) to the organ region (block L). An adapter is in the window if the window covers
    at least a fraction `overlap` (epsilon) of it along each axis.
    G = 4, L = 12, epsilon = 1/3: blocks 1-4 centre region (4 adapters), 5-8 a 3 x 3 block (9), 9-12 organ region (4).
    stages = (s1, s2) (ablation): fixed windows, blocks 1..s1 the centre region, s1+1..L the organ region
    (e.g. (0, 12) = organ region in every block)."""
    def __init__(self, grid, n_layers, stages=None, overlap=1/3):
        assert grid % 2 == 0, "the grid needs an even size (2 x 2 organ regions)"
        self.G, self.M, self.L = grid, grid * grid, n_layers
        self.overlap = float(overlap)
        self.stages = None if stages is None else (int(stages[0]), int(stages[1]))
        self.n_regions = (grid // 2) ** 2                                      # organs the grid can hold
        self.c0 = (grid // 2 - 1, grid // 2 - 1)                               # top-left of the centre region

    def _origin(self, organ):
        br, bc = divmod(int(organ), self.G // 2)
        return 2 * br, 2 * bc

    def _covered(self, top, left):
        """Adapters covered by a 2 x 2 window with top-left corner at (top, left) (possibly fractional)."""
        def axis(a):
            out = []
            for r in range(self.G):
                ov = min(r + 1, a + 2) - max(r, a)                      # overlap of cell [r, r+1) with [a, a+2)
                if ov > 1e-9 and ov >= self.overlap - 1e-9:
                    out.append(r)
            return out
        return [r * self.G + c for r in axis(top) for c in axis(left)]

    def __call__(self, organ):
        """(L, M) bool: the adapters in the window of each block."""
        assert organ < self.n_regions, f"organ {organ}: the {self.G}x{self.G} grid holds {self.n_regions} organs"
        W = torch.zeros(self.L, self.M, dtype=torch.bool)
        o = self._origin(organ)
        for l in range(self.L):
            if self.stages is not None:                                 # fixed windows (ablation)
                f = 0.0 if l < self.stages[0] else 1.0
            else:
                f = l / max(self.L - 1, 1)                              # tau_l
            W[l, self._covered(self.c0[0] + f * (o[0] - self.c0[0]), self.c0[1] + f * (o[1] - self.c0[1]))] = True
        return W


# -------------------------------------------------------------------
# Adapter pool (dense dispatch of the top-k mixture)
# -------------------------------------------------------------------
class ExpertPool(nn.Module):
    """Pool of bottleneck adapters shared by all blocks; top-k mixture with renormalised weights, computed densely
    (all adapters as two stacked matmuls, each adapter's hidden units scaled by its weight: unselected adapters
    contribute 0 in the forward and backward pass)."""
    def __init__(self, experts):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self._used = None                           # (M,) bool: experts selected since drop_unused_grads()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_used"] = None
        return state

    @torch.no_grad()
    def drop_unused_grads(self):
        """After backward: experts no token selected get grad None (AdamW leaves them untouched)."""
        if self._used is None:
            return
        for used, ex in zip(self._used.tolist(), self.experts):
            if not used:
                for p in ex.parameters():
                    p.grad = None
        self._used = None

    def forward(self, x, probs, top_k):
        top_w, top_i = probs.topk(top_k, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True).clamp_min(1e-9)
        N, M = x.shape[0], len(self.experts)
        if torch.is_grad_enabled():
            used = torch.zeros(M, dtype=torch.bool, device=x.device).scatter_(0, top_i.reshape(-1), True)
            self._used = used if self._used is None else self._used | used
        w = torch.zeros(N, M, dtype=top_w.dtype, device=x.device).scatter(1, top_i, top_w)
        Wd = torch.cat([ex.down.weight for ex in self.experts], 0)
        bd = torch.cat([ex.down.bias for ex in self.experts], 0)
        Wu = torch.cat([ex.up.weight for ex in self.experts], 1)
        bu = torch.stack([ex.up.bias for ex in self.experts], 0)
        h = F.relu(F.linear(x, Wd, bd)).view(N, M, -1) * w.unsqueeze(-1)
        return F.linear(h.reshape(N, -1), Wu) + w @ bu


# -------------------------------------------------------------------
# Encoder blocks
# -------------------------------------------------------------------
class _FrozenBlock(nn.Module):
    """Shared helpers: frozen attention (keeps SAM's window partition) without grad through the frozen weights."""
    def _frozen_ctx(self):
        return contextlib.nullcontext() if self.grad_through_frozen else torch.no_grad()

    def _attn(self, x_norm):
        blk = self.original_block
        with self._frozen_ctx():
            if blk.window_size > 0:
                H, W = x_norm.shape[1], x_norm.shape[2]
                xw, pad_hw = window_partition(x_norm, blk.window_size)
                out = window_unpartition(blk.attn(xw), blk.window_size, pad_hw, (H, W))
            else:
                out = blk.attn(x_norm)
        return out


class FedSHABlock(_FrozenBlock):
    """Frozen SAM block + gated mixture of the shared adapter grid next to the attention and the MLP branch.
    Branch output: (1 + gamma) * sum_{e in top-k of the window} w_e A_e(x) + beta."""
    def __init__(self, original_block, hidden, n_adapters, top_k=4, use_checkpoint=False, grad_through_frozen=False):
        super().__init__()
        self.original_block = original_block
        self.top_k = top_k
        self.use_checkpoint, self.grad_through_frozen = use_checkpoint, grad_through_frozen
        self.gate_attn = nn.Linear(hidden, n_adapters)
        self.gate_mlp = nn.Linear(hidden, n_adapters)
        for g in (self.gate_attn, self.gate_mlp):
            nn.init.normal_(g.weight, std=0.01)
            nn.init.zeros_(g.bias)
        self.mod_gamma_attn = nn.Parameter(torch.zeros(hidden))
        self.mod_beta_attn = nn.Parameter(torch.zeros(hidden))
        self.mod_gamma_mlp = nn.Parameter(torch.zeros(hidden))
        self.mod_beta_mlp = nn.Parameter(torch.zeros(hidden))
        for p in self.original_block.parameters():
            p.requires_grad = False

    def _mixture(self, x, gate, pool, gamma, beta, window):
        B, H, W, D = x.shape
        flat = x.reshape(-1, D)
        logits = gate(flat)
        if window is not None:                           # the gate decides the weights inside the window
            logits = logits.masked_fill(~window, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        k = self.top_k if window is None else min(self.top_k, int(window.sum()))
        out = pool(flat, probs, k).view(B, H, W, D)
        return (1 + gamma) * out + beta, probs.view(B, H, W, -1)

    def forward(self, x, pool, window=None):
        blk = self.original_block

        def _forward(x):
            n1 = blk.norm1(x)
            a, p_attn = self._mixture(n1, self.gate_attn, pool, self.mod_gamma_attn, self.mod_beta_attn, window)
            x = x + self._attn(n1) + a
            n2 = blk.norm2(x)
            with self._frozen_ctx():
                mlp_out = blk.mlp(n2)
            m, p_mlp = self._mixture(n2, self.gate_mlp, pool, self.mod_gamma_mlp, self.mod_beta_mlp, window)
            return x + mlp_out + m, p_attn, p_mlp

        if self.use_checkpoint and self.training:
            return checkpoint(_forward, x, use_reentrant=False)
        return _forward(x)


class MSABlock(_FrozenBlock):
    """Frozen SAM block + one dense adapter next to the attention and one next to the MLP branch (MSA)."""
    def __init__(self, original_block, hidden, adapter_reduction=16, use_checkpoint=False, grad_through_frozen=False):
        super().__init__()
        self.original_block = original_block
        self.use_checkpoint, self.grad_through_frozen = use_checkpoint, grad_through_frozen
        self.msa_adapter_attn = Adapter(hidden, adapter_reduction)
        self.msa_adapter_mlp = Adapter(hidden, adapter_reduction)
        for p in self.original_block.parameters():
            p.requires_grad = False

    def forward(self, x):
        blk = self.original_block

        def _forward(x):
            n1 = blk.norm1(x)
            x = x + self._attn(n1) + self.msa_adapter_attn(n1)
            n2 = blk.norm2(x)
            with self._frozen_ctx():
                mlp_out = blk.mlp(n2)
            return x + mlp_out + self.msa_adapter_mlp(n2)

        if self.use_checkpoint and self.training:
            return checkpoint(_forward, x, use_reentrant=False)
        return _forward(x)


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
class SegModel(nn.Module):
    """Frozen MedSAM + its mask decoder without prompts. Subclasses define the encoder adaptation."""
    def _init_sam(self, checkpoint_path, img_size):
        self.sam = adapt_sam_resolution(sam_model_registry["vit_b"](checkpoint=checkpoint_path or None), img_size)
        self.n_seg = 1

    def _set_trainable(self):
        for name, p in self.named_parameters():
            p.requires_grad = param_group(name)[0] != "frozen"

    def configure_heads(self, n_seg=1):
        assert 1 <= n_seg <= self.sam.mask_decoder.num_mask_tokens, f"n_seg={n_seg}: SAM has 4 mask tokens"
        self.n_seg = n_seg

    def encode(self, images):
        raise NotImplementedError

    def decode(self, emb):
        """MedSAM mask decoder without prompts -> mask logits of the client's n_seg slots at img_size / 4."""
        md, pe = self.sam.mask_decoder, self.sam.prompt_encoder
        B, C, H, W = emb.shape
        tokens = torch.cat([md.iou_token.weight, md.mask_tokens.weight], 0)[None].expand(B, -1, -1)
        src = emb + pe.no_mask_embed.weight.reshape(1, -1, 1, 1)
        pos = pe.get_dense_pe().expand(B, -1, -1, -1)
        hs, src = md.transformer(src, pos, tokens)                     # hs (B, 5, C), src (B, HW, C)
        up = md.output_upscaling(src.transpose(1, 2).reshape(B, C, H, W))
        hyper = torch.stack([md.output_hypernetworks_mlps[i](hs[:, 1 + i]) for i in range(self.n_seg)], 1)
        b, c, h, w = up.shape
        return (hyper @ up.view(b, c, h * w)).view(b, -1, h, w)

    def forward(self, images):
        emb = self.encode(images)
        b = self._parameters.get("dbe_bias")                       # FedDBE: emb + local bias before the decoder
        return {"seg": self.decode(emb if b is None else emb + b.view(1, -1, 1, 1)), "emb": emb}

    def enable_dbe(self):
        """FedDBE: personalized representation bias memory (PRBM), one learnable vector per neck channel."""
        self.dbe_bias = nn.Parameter(torch.zeros(self.sam.image_encoder.neck[0].out_channels))

    # no-ops for the baselines
    def drop_unused_expert_grads(self):
        pass

    def clear_routing(self):
        pass


class MSAModel(SegModel):
    """Baseline network: frozen MedSAM + dense MSA adapters (width 48) + MedSAM decoder."""
    def __init__(self, checkpoint_path="./medsam_vit_b.pth", adapter_reduction=16, img_size=512,
                 use_checkpoint=False, grad_through_frozen=False, **unused):
        super().__init__()
        self._init_sam(checkpoint_path, img_size)
        enc = self.sam.image_encoder
        hidden = enc.blocks[0].mlp.lin1.in_features
        for i, blk in enumerate(enc.blocks):
            enc.blocks[i] = MSABlock(blk, hidden, adapter_reduction, use_checkpoint, grad_through_frozen)
        self._set_trainable()

    def encode(self, images):
        enc = self.sam.image_encoder
        x = enc.patch_embed(images)
        if enc.pos_embed is not None:
            x = x + enc.pos_embed
        for blk in enc.blocks:
            x = blk(x)
        return enc.neck(x.permute(0, 3, 1, 2))


class MoSABlock(_FrozenBlock):
    """FedMoSA block: frozen SAM block + modality-specialised adapter (one per modality, hard-routed by the batch's
    modality) + shared adapter, next to the attention and to the MLP branch:
        x = x + attn(n1) + A^m_attn(n1) + S_attn(n1);   x = x + mlp(n2) + A^m_mlp(n2) + S_mlp(n2)
    With collect on, the mean of each shared adapter's output over batch and tokens (with gradient) is kept as the
    batch prototype of that adapter."""
    def __init__(self, original_block, hidden, modalities, adapter_reduction=16, use_checkpoint=False,
                 grad_through_frozen=False):
        super().__init__()
        self.original_block = original_block
        self.use_checkpoint, self.grad_through_frozen = use_checkpoint, grad_through_frozen
        self.mosa_spec_attn = nn.ModuleDict({m: Adapter(hidden, adapter_reduction) for m in modalities})
        self.mosa_spec_mlp = nn.ModuleDict({m: Adapter(hidden, adapter_reduction) for m in modalities})
        self.mosa_shared_attn = Adapter(hidden, adapter_reduction)
        self.mosa_shared_mlp = Adapter(hidden, adapter_reduction)
        self.modality = modalities[0]
        self.collect = False
        self.proto = {}                               # {"attn": (C,), "mlp": (C,)} of the last forward
        for p in self.original_block.parameters():
            p.requires_grad = False

    def forward(self, x):
        blk, m = self.original_block, self.modality
        n1 = blk.norm1(x)
        s_attn = self.mosa_shared_attn(n1)
        x = x + self._attn(n1) + self.mosa_spec_attn[m](n1) + s_attn
        n2 = blk.norm2(x)
        with self._frozen_ctx():
            mlp_out = blk.mlp(n2)
        s_mlp = self.mosa_shared_mlp(n2)
        x = x + mlp_out + self.mosa_spec_mlp[m](n2) + s_mlp
        if self.collect:
            self.proto = {"attn": s_attn.flatten(0, 2).mean(0), "mlp": s_mlp.flatten(0, 2).mean(0)}
        return x


class MoSAModel(SegModel):
    """FedMoSA on the common base network: frozen MedSAM + MoSA blocks + MedSAM decoder."""
    def __init__(self, checkpoint_path="./medsam_vit_b.pth", modalities=("CT", "MR"), adapter_reduction=16,
                 img_size=512, use_checkpoint=False, grad_through_frozen=False, **unused):
        super().__init__()
        self._init_sam(checkpoint_path, img_size)
        enc = self.sam.image_encoder
        hidden = enc.blocks[0].mlp.lin1.in_features
        self.modalities = list(modalities)
        for i, blk in enumerate(enc.blocks):
            enc.blocks[i] = MoSABlock(blk, hidden, self.modalities, adapter_reduction, use_checkpoint,
                                      grad_through_frozen)
        self._set_trainable()

    def set_modality(self, modality):
        assert modality in self.modalities, f"unknown modality {modality}, expected {self.modalities}"
        for blk in self.sam.image_encoder.blocks:
            blk.modality = modality

    def set_collect(self, flag):
        for blk in self.sam.image_encoder.blocks:
            blk.collect = flag
            blk.proto = {}

    def batch_prototypes(self):
        """[(attn, mlp)] per block: shared-adapter output means of the last forward (with gradient)."""
        return [(b.proto.get("attn"), b.proto.get("mlp")) for b in self.sam.image_encoder.blocks]

    encode = MSAModel.encode


class MENUBlock(_FrozenBlock):
    """Frozen SAM block + one pair of MSA adapters (attention, MLP) per organ; forward(x, m) uses organ m's pair."""
    def __init__(self, original_block, hidden, n_organs, adapter_reduction=16, grad_through_frozen=False):
        super().__init__()
        self.original_block = original_block
        self.grad_through_frozen = grad_through_frozen
        self.menu_adapters = nn.ModuleList([nn.ModuleDict({"attn": Adapter(hidden, adapter_reduction),
                                                           "mlp": Adapter(hidden, adapter_reduction)})
                                            for _ in range(n_organs)])
        for p in self.original_block.parameters():
            p.requires_grad = False

    def forward(self, x, m):
        blk, ad = self.original_block, self.menu_adapters[m]
        n1 = blk.norm1(x)
        x = x + self._attn(n1) + ad["attn"](n1)
        n2 = blk.norm2(x)
        with self._frozen_ctx():
            mlp_out = blk.mlp(n2)
        return x + mlp_out + ad["mlp"](n2)


class MENUModel(SegModel):
    """Fed-MENU (multi-encoding U-Net) ported to the frozen MedSAM.
    Sub-encoder g_m = frozen ViT + organ m's adapters (one per organ of the federation). Every image goes through
    all M sub-encoders; their neck features are concatenated channel-wise and fused (1x1 conv, initialised to the
    mean, so training starts at MedSAM) before the MedSAM decoder: F(x) = f( concat_m g_m(x) ).
    A client tunes only the sub-encoder of its own organ (the others run without gradient, as in Fed-MENU).
    AGD (auxiliary generic decoder) h: on the client's own sub-encoder features, predicts an organ-agnostic binary
    mask: two 3x3 conv + InstanceNorm + LeakyReLU, then a 1x1 conv with 2 output channels (softmax)."""
    def __init__(self, checkpoint_path="./medsam_vit_b.pth", n_organs=4, adapter_reduction=16, img_size=512,
                 grad_through_frozen=False, **unused):
        super().__init__()
        self._init_sam(checkpoint_path, img_size)
        enc = self.sam.image_encoder
        hidden = enc.blocks[0].mlp.lin1.in_features
        C = self.sam.prompt_encoder.embed_dim
        self.n_organs, self.own = int(n_organs), None
        for i, blk in enumerate(enc.blocks):
            enc.blocks[i] = MENUBlock(blk, hidden, self.n_organs, adapter_reduction, grad_through_frozen)
        self.menu_fuse = nn.Conv2d(C * self.n_organs, C, 1)
        with torch.no_grad():                                     # start as the mean of the sub-encoders (= MedSAM)
            self.menu_fuse.weight.zero_()
            self.menu_fuse.bias.zero_()
            for m in range(self.n_organs):
                self.menu_fuse.weight[:, m * C:(m + 1) * C, 0, 0] = torch.eye(C) / self.n_organs
        self.menu_agd = nn.Sequential(
            nn.Conv2d(C, 64, 3, padding=1), nn.InstanceNorm2d(64, affine=True), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.InstanceNorm2d(64, affine=True), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(64, 2, 1))
        self._own_emb = None
        self._set_trainable()

    def set_organ(self, m):
        self.own = int(m)

    def _sub_encoder(self, images, m):
        enc = self.sam.image_encoder
        x = enc.patch_embed(images)
        if enc.pos_embed is not None:
            x = x + enc.pos_embed
        for blk in enc.blocks:
            x = blk(x, m)
        return enc.neck(x.permute(0, 3, 1, 2))

    def encode(self, images):
        embs = []
        for m in range(self.n_organs):
            if m == self.own and torch.is_grad_enabled():
                e = self._sub_encoder(images, m)
                self._own_emb = e
            else:                                                 # other organs' sub-encoders: not tuned here
                with torch.no_grad():
                    e = self._sub_encoder(images, m)
                if m == self.own:
                    self._own_emb = e
            embs.append(e)
        return self.menu_fuse(torch.cat(embs, 1))

    def forward(self, images):
        emb = self.encode(images)
        out = {"seg": self.decode(emb), "emb": emb}
        if self._own_emb is not None:
            out["agd"] = self.menu_agd(self._own_emb)             # (B, 2, g, g) organ-agnostic mask logits
        return out


class FedSHA(SegModel):
    """Ours: shared adapter grid with a depth-dependent sliding window and gated top-k mixture.
    hard_window=False (ablation): no window, the gate chooses the top-k of the whole grid in every block."""
    def __init__(self, checkpoint_path="./medsam_vit_b.pth", grid=4, top_k=4, win_stages=None, win_overlap=1/3,
                 hard_window=True, adapter_reduction=16, img_size=512, use_checkpoint=False,
                 grad_through_frozen=False, **unused):
        super().__init__()
        self._init_sam(checkpoint_path, img_size)
        enc = self.sam.image_encoder
        hidden = enc.blocks[0].mlp.lin1.in_features
        self.G, self.M = grid, grid * grid
        self.top_k, self.hard_window = top_k, bool(hard_window)
        self.anatomy_pool = ExpertPool([Adapter(hidden, adapter_reduction) for _ in range(self.M)])   # the grid
        for i, blk in enumerate(enc.blocks):
            enc.blocks[i] = FedSHABlock(blk, hidden, self.M, top_k, use_checkpoint, grad_through_frozen)
        self.n_layers = len(enc.blocks)
        self.window_generator = SlidingWindow(grid, self.n_layers, win_stages, win_overlap)
        self.register_buffer("anatomy_mask", None)           # (L, M) bool windows, set by set_organ()
        self.placed = False
        self._routing = []
        self._set_trainable()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_routing"] = []
        return state

    def clear_routing(self):
        self._routing = []

    def set_organ(self, organ):
        """Build this client's windows locally from its organ region (index sent by the server)."""
        dev = self.anatomy_pool.experts[0].up.weight.device
        self.anatomy_mask = self.window_generator(organ).to(dev)
        self.placed = True
        am, segs, start = self.anatomy_mask, [], 0
        for l in range(1, self.n_layers + 1):
            if l == self.n_layers or not torch.equal(am[l], am[start]):
                segs.append(f"blocks {start + 1}-{l} {am[start].nonzero().flatten().tolist()}")
                start = l
        print(f"  windows (top-{self.top_k}): " + " | ".join(segs))

    def in_window(self):
        """(M,) bool: adapters in at least one of the client's windows (all adapters without a window)."""
        if not self.hard_window or self.anatomy_mask is None:
            return torch.ones(self.M, dtype=torch.bool)
        return self.anatomy_mask.any(0).cpu()

    def encode(self, images):
        enc = self.sam.image_encoder
        x = enc.patch_embed(images)
        if enc.pos_embed is not None:
            x = x + enc.pos_embed
        win = self.hard_window and self.anatomy_mask is not None
        self._routing = []
        for l, blk in enumerate(enc.blocks):
            x, pa, pm = blk(x, self.anatomy_pool, self.anatomy_mask[l] if win else None)
            if self.training:                                 # gate probabilities (gate mass for aggregation)
                self._routing.append(0.5 * (pa.detach().mean(dim=(0, 1, 2)) + pm.detach().mean(dim=(0, 1, 2))))
        return enc.neck(x.permute(0, 3, 1, 2))

    def drop_unused_expert_grads(self):
        self.anatomy_pool.drop_unused_grads()

    @torch.no_grad()
    def block_routing(self):
        """(L, M) mean gate probability per block of the last training forward pass (None if none)."""
        return torch.stack(self._routing).float().cpu() if self._routing else None

    def routing_report(self, R):
        lines = ["  block | top-4 adapters   | mass in window"]
        for l, r in enumerate(R):
            r = r / r.sum()
            win = self.anatomy_mask[l].cpu() if self.anatomy_mask is not None else torch.ones_like(r, dtype=torch.bool)
            lines.append(f"  {l + 1:5d} | {str(sorted(r.topk(4).indices.tolist())):16s} | {float(r[win].sum()):.2f}")
        return "\n".join(lines)
