"""
Arsitektur model untuk Oil Palm Ripeness Classification.

Backbone: ResNet-18 (torchvision) + custom classifier head.
Struktur ini direkonstruksi berdasarkan key-key state_dict di dalam
'best_model_multiobjective.pth' (hasil training hybrid GA-PSO-SA
multi-objective). Confirmed cocok 1:1 dengan strict state_dict loading
(tidak ada missing/unexpected keys).

Catatan penting:
- File .pth yang diberikan HANYA berisi bobot (state_dict), bukan definisi
  kelas model. Karena itu arsitektur ini WAJIB sama persis dengan yang
  dipakai saat training, atau load_state_dict akan gagal / salah.
- Jika model asli Anda punya modul Self-Adaptive Dynamic Attention (SADA)
  yang terpisah dari backbone ResNet18 standar, TAPI bobotnya tidak
  muncul di state_dict ini, kemungkinan attention module sudah menyatu
  dalam salah satu blok (mis. built into forward pass tanpa parameter
  terlatih tambahan, atau memang belum tersimpan di checkpoint ini).
  Jika hasil prediksi terasa tidak akurat, kirim ulang kode training
  model aslinya agar arsitektur bisa disesuaikan persis.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18

# Urutan kelas HARUS sama dengan urutan saat training (index 0..3)
#
# PENTING: jika training memakai torchvision.datasets.ImageFolder, urutan
# index kelas ditentukan SECARA ALFABETIS dari nama folder, bukan urutan
# logis kematangan. Untuk nama folder Unripe/Underripe/Ripe/Overripe,
# urutan alfabetisnya adalah:
#   0 = Overripe, 1 = Ripe, 2 = Underripe, 3 = Unripe
#
# Ini BERBEDA dari urutan logis (Unripe->Underripe->Ripe->Overripe) yang
# dipakai di versi awal file ini -- itu penyebab paling mungkin kenapa
# "Ripe" terdeteksi sebagai "Underripe" (index-nya tertukar).
#
# VERIFIKASI PALING AKURAT: cek `dataset.class_to_idx` atau
# `train_dataset.classes` dari kode training asli Anda, lalu sesuaikan
# urutan di bawah ini agar 100% cocok.
CLASS_NAMES = ["Overripe", "Ripe", "Underripe", "Unripe"]

IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(num_classes: int = 4) -> nn.Module:
    """Bangun arsitektur ResNet18 + custom head sesuai state_dict checkpoint."""
    model = resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),                  # fc.0
        nn.Linear(512, 512),              # fc.1
        nn.ReLU(inplace=True),            # fc.2
        nn.BatchNorm1d(512),              # fc.3
        nn.Dropout(0.3),                  # fc.4
        nn.Linear(512, num_classes),      # fc.5
    )
    return model


def load_model(weights_path: str, device: str = "cpu") -> nn.Module:
    """Load model + bobot dari file .pth (state_dict)."""
    model = build_model(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
