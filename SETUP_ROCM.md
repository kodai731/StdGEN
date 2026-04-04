# StdGEN ROCm (gfx1100) 環境構築手順

## 前提環境

- WSL2 + RX 7900 XTX (gfx1100)
- ROCm 7.2
- uv (パッケージマネージャ)
- CharacterGenと同じRadeonリポジトリのPyTorchを使用

## 1. venv作成

StdGENはPython 3.9を指定しているが、ROCm 7.2のPyTorchはPython 3.12を使用。
CharacterGenと統一してPython 3.12を使用する。

```bash
cd /home/kodai/Projects/AnimationModelTraining/third_party/StdGEN
uv venv --python 3.12 .venv
source .venv/bin/activate
```

## 2. PyTorch (ROCm 7.2) インストール

公式PyPI indexにはROCm 7.2版が無い。Radeonリポジトリから直接インストール:

```bash
uv pip install \
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl" \
    "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
```

## 3. xformers

ROCm 7.2用のxformers wheelは無い。スキップ可能。
StdGENはxformersが無い場合PyTorch native attentionにフォールバックする。

## 4. torch-scatter / pytorch3d

いずれもStage 3 (Refine) でのみ使用。ROCm 7.2でのビルドに失敗するためスキップ。
Stage 3に到達した段階で別途対応する。

- torch-scatter → `torch.scatter_reduce` で代替可能
- pytorch3d → ソースビルド時に `--no-build-isolation` が必要

## 5. 依存パッケージ

```bash
uv pip install \
    omegaconf rm_anime_bg "diffusers==0.27.2" "transformers==4.42.4" \
    einops "huggingface_hub==0.25.0" opencv-python accelerate \
    matplotlib kornia imageio imageio-ffmpeg \
    xatlas trimesh rembg onnxruntime scikit-learn \
    pygltflib pymeshlab pytorch_lightning mcubes

uv pip install git+https://github.com/Baijiong-Lin/LoRA-Torch
uv pip install git+https://github.com/facebookresearch/segment-anything.git
```

## 6. nvdiffrast 対応（ROCm非互換）

nvdiffrastはCUDA専用。インストールしない。
CharacterGenと同じスタブ方式でローカルモジュールを作成して対応する。

Stage 1.1, 1.2 はnvdiffrastを使わないため、スタブ作成前でも動作確認可能。

## 7. モデルダウンロード

HuggingFaceリポジトリ `hyz317/StdGEN` にはfp16版・旧版が含まれる。
最小限のファイルのみダウンロードする。

### Stage 1.1 (Canonicalize) — 約10.7GB

```bash
.venv/bin/python -c "
from huggingface_hub import hf_hub_download

repo = 'hyz317/StdGEN'
files = [
    'StdGEN-canonicalize-1024/model_index.json',
    'StdGEN-canonicalize-1024/feature_extractor/preprocessor_config.json',
    'StdGEN-canonicalize-1024/image_encoder/config.json',
    'StdGEN-canonicalize-1024/image_encoder/pytorch_model.bin',
    'StdGEN-canonicalize-1024/ref_unet/config.json',
    'StdGEN-canonicalize-1024/ref_unet/diffusion_pytorch_model.safetensors',
    'StdGEN-canonicalize-1024/text_encoder/config.json',
    'StdGEN-canonicalize-1024/text_encoder/model.safetensors',
    'StdGEN-canonicalize-1024/unet/config.json',
    'StdGEN-canonicalize-1024/unet/diffusion_pytorch_model.safetensors',
    'StdGEN-canonicalize-1024/vae/config.json',
    'StdGEN-canonicalize-1024/vae/diffusion_pytorch_model.safetensors',
    'StdGEN-canonicalize-1024/tokenizer/merges.txt',
    'StdGEN-canonicalize-1024/tokenizer/special_tokens_map.json',
    'StdGEN-canonicalize-1024/tokenizer/tokenizer_config.json',
    'StdGEN-canonicalize-1024/tokenizer/vocab.json',
    'StdGEN-canonicalize-1024/scheduler-zerosnr/scheduler_config.json',
]
for f in files:
    print(f'Downloading {f}...')
    hf_hub_download(repo, f, local_dir='./ckpt')
print('Done!')
"
```

除外したファイル:
- `text_encoder/pytorch_model.bin` — safetensors版と重複
- `text_encoder/model.fp16.safetensors` — fp16版不要
- `text_encoder/pytorch_model.fp16.bin` — fp16版不要
- `vae/diffusion_pytorch_model.bin` — safetensors版と重複
- `vae/diffusion_pytorch_model.fp16.*` — fp16版不要

### Stage 1.2 (Multiview) — 約3.8GB

```bash
.venv/bin/python -c "
from huggingface_hub import hf_hub_download

repo = 'hyz317/StdGEN'
files = [
    'StdGEN-multiview-1024/model_index.json',
    'StdGEN-multiview-1024/feature_extractor/preprocessor_config.json',
    'StdGEN-multiview-1024/image_encoder/config.json',
    'StdGEN-multiview-1024/image_encoder/model.safetensors',
    'StdGEN-multiview-1024/image_noising_scheduler/scheduler_config.json',
    'StdGEN-multiview-1024/image_normalizer/config.json',
    'StdGEN-multiview-1024/image_normalizer/diffusion_pytorch_model.safetensors',
    'StdGEN-multiview-1024/scheduler/scheduler_config.json',
    'StdGEN-multiview-1024/text_encoder/config.json',
    'StdGEN-multiview-1024/text_encoder/model.safetensors',
    'StdGEN-multiview-1024/tokenizer/merges.txt',
    'StdGEN-multiview-1024/tokenizer/special_tokens_map.json',
    'StdGEN-multiview-1024/tokenizer/tokenizer_config.json',
    'StdGEN-multiview-1024/tokenizer/vocab.json',
    'StdGEN-multiview-1024/unet/config.json',
    'StdGEN-multiview-1024/unet/diffusion_pytorch_model.safetensors',
    'StdGEN-multiview-1024/vae/config.json',
    'StdGEN-multiview-1024/vae/diffusion_pytorch_model.safetensors',
]
for f in files:
    print(f'Downloading {f}...')
    hf_hub_download(repo, f, local_dir='./ckpt')
print('Done!')
"
```

除外したファイル:
- `unet-old/` — 旧版UNet不要

### Stage 2 (S-LRM) — 約1.6GB

```bash
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('hyz317/StdGEN', 'StdGEN-mesh-slrm.pth', local_dir='./ckpt')
print('Done!')
"
```

DINO ViT-B16 (`facebook/dino-vitb16`) は推論時にHuggingFaceから自動ダウンロードされる。

### Stage 3 (Refine) — 約2.5GB

```bash
wget -P ./ckpt/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## nvdiffrast 使用箇所（要置換）

### 使用パターン

全てのnvdiffrast呼び出しは3つの関数に集約される:

| 関数 | 用途 | 呼び出し元 |
|------|------|-----------|
| `dr.rasterize()` | メッシュラスタライズ | slrm/, refine/ |
| `dr.interpolate()` | 頂点属性補間 | slrm/, refine/ |
| `dr.antialias()` | アンチエイリアス | slrm/, refine/ |

### ファイル別の使用箇所

| ファイル | 使用内容 | ステージ |
|---------|---------|---------|
| `slrm/models/lrm.py` | テクスチャマップUV展開 | Stage 2 |
| `slrm/models/lrm_mesh.py` | テクスチャマップUV展開 | Stage 2 |
| `slrm/utils/mesh_util.py` | xatlas_uvmap内のラスタライズ | Stage 2 |
| `slrm/models/geometry/render/neural_render.py` | 法線レンダリング + アンチエイリアス | Stage 2 |
| `slrm/models/geometry/rep_3d/extract_texture_map.py` | テクスチャ抽出 | Stage 2 |
| `refine/render.py` | 法線レンダリング + NormalsRenderer | Stage 3 |
| `refine/func.py` | メッシュリファインメント用ラスタライズ | Stage 3 |

### RasterizeCudaContext 生成箇所（6箇所）

1. `refine/render.py:17` — グローバル変数
2. `refine/render.py:21` — NormalsRenderer クラス属性
3. `refine/func.py:137` — RefineRenderer.__init__
4. `slrm/models/lrm_mesh.py:596` — extract_mesh
5. `slrm/models/lrm.py:226` — extract_mesh
6. `slrm/models/geometry/render/neural_render.py:74` — NeuralRender.__init__

### 置換戦略

CharacterGenで実施済みのアプローチを流用:
1. `nvdiffrast/` ディレクトリにスタブモジュールを作成
2. `dr.rasterize` → PyTorch純正テンソル操作 or PyTorch3D rasterizer
3. `dr.interpolate` → 頂点属性補間のPyTorch実装
4. `dr.antialias` → パススルー（省略可能）

## 動作確認順序

1. **Stage 1.1 (Canonicalize)** — nvdiffrast不要、拡散モデルのみ
2. **Stage 1.2 (Multiview)** — nvdiffrast不要、拡散モデルのみ
3. **Stage 2 (S-LRM)** — nvdiffrastスタブ作成後に動作確認
4. **Stage 3 (Refine)** — nvdiffrast + pytorch3d + torch-scatter 対応後に動作確認

## ライセンス

| 項目 | 内容 |
|------|------|
| コード | Apache 2.0（商用利用可、帰属表示必須） |
| プリトレインモデル | **研究目的のみ**（README Disclaimerで明記） |
| nvdiffrast | Nvidia Source Code License（商用要確認） |
| SAM weights | Apache 2.0 |
| VRoid学習データ | 生データ再配布不可 |
