"""Reference LSTD-Mamba configuration (paper Section IV-A-3).

This is the *student-only* configuration: it contains no teacher and no
distillation loss. The reported model has 1.98M parameters and 77.94G FLOPs,
and is trained for 100 epochs with AdamW at an initial learning rate of
1.5e-4.

Training protocol from the paper:

* 100 epochs, batch size 8 per GPU, global batch size 16 on two GPUs;
* AdamW, weight decay 0.01, dynamic-loss-scale mixed precision;
* gradient clipping at 1.0;
* 500-iteration linear warm-up, then cosine annealing to 1e-5;
* random horizontal/vertical flips and rotations, modality-specific
  normalization and resizing.

Inputs: six ten-band Sentinel-2 observations at 192x192 and one 218-band
EnMAP cube at 64x64. The four label levels contain 7, 37, 83 and 102 classes.
"""

# Register the LSTD-Mamba modules with the mmseg registry before the model is
# built. H2Crop must be importable (see README, step "Upstream H2Crop").
custom_imports = dict(
    imports=["lstd_mamba", "lstd_mamba.models"], allow_failed_imports=False
)

# --- Data ------------------------------------------------------------------
data_root = "data/h2crop"
data_list_root = "data/h2crop/data_list"
train_list = data_list_root + "/train.txt"
val_list = data_list_root + "/val.txt"
test_list = data_list_root + "/test.txt"

img_size = 192
hyper_img_size = 64
input_seq_len = 6
num_frames = 6

# Historical crop priors are enabled in the main paper configuration.
with_priors = True
with_enmap = True

levels = (
    ("level1", 7),
    ("level2", 37),
    ("level3", 83),
    ("level4", 102),
)

# --- Model -----------------------------------------------------------------
model = dict(
    type="LSTDMamba",
    with_priors=with_priors,
    with_enmap=with_enmap,
    levels=levels,
    # Sentinel-2 temporal branch: three stages of widths (64, 128, 256), one
    # block per stage. LST occupies the shallow stages and BTM the final one.
    s2_encoder=dict(
        in_channels=10,
        stage_channels=(64, 128, 256),
        depths=(1, 1, 1),
        btm_stage=2,
        stem_channels=64,
        d_state=8,
        d_conv=3,
        expand=2,
        drop_path_rate=0.1,
        bidirectional=True,
    ),
    # EnMAP spectral branch: full-resolution local projection plus two BSM
    # blocks on a stride-4 pooled grid with a 32-channel token representation.
    enmap_encoder=dict(
        in_bands=218,
        spectral_channels=32,
        out_channels=64,
        depth=2,
        spatial_pool=4,
        spectral_stride=2,
        d_state=8,
        d_conv=3,
        expand=2,
        drop_path_rate=0.05,
        bidirectional=True,
    ),
    # Lightweight top-down decoder with additive multimodal fusion (Eq. 13).
    neck=dict(
        embed_dim=256,
        in_feature_key=("S2",),
        fusion_channels=(256, 128, 128),
        out_channels=128,
        stage_channels=(64, 128, 256),
        hyper_embed_neck=dict(in_channels=64, in_key="EnMAP"),
    ),
    # Prior-aware four-level cascade head (Eq. 14).
    head=dict(
        embed_dim=128,
        with_priors=with_priors,
    ),
    # Effective-number weighted cross-entropy per level (Eqs. 15-16).
    loss=dict(
        type="EffectiveNumberCrossEntropyLoss",
        mode="effective",
        class_counts_path="lstd_mamba/resources/h2crop_train_class_counts.json",
        beta=0.9999,
        ignore_index=255,
    ),
)

# --- Training schedule -----------------------------------------------------
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100, val_interval=1)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=1.5e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    loss_scale="dynamic",
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(type="CosineAnnealingLR", eta_min=1e-5, by_epoch=False, begin=500),
]

randomness = dict(seed=0, deterministic=False, diff_rank_seed=False)
