"""
Parquet-backed Dataset. Reads image bytes and labels from a Parquet file.
"""
import io
import time
from typing import Optional, Callable

import torch
from PIL import Image
import torchvision.transforms as T
import pyarrow.parquet as pq


class ParquetDataset:
    def __init__(self, parquet_path: str, max_samples: int = None,
                 transform: Optional[Callable] = None):
        self.parquet_path = parquet_path
        self.transform = transform

        self._table = pq.read_table(parquet_path)
        self._length = len(self._table)
        if max_samples:
            self._length = min(self._length, max_samples)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        # Stage 1: I/O - reading raw bytes from Parquet table
        t0 = time.perf_counter()
        row = self._table.slice(index, 1).to_pylist()[0]
        img_bytes = row['image']
        label = row['label']
        t1 = time.perf_counter()

        # Stage 2: JPEG decoding
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        t2 = time.perf_counter()

        # Stage 3: Preprocessing (resize + ToTensor)
        img, label = self.preprocess(img, label)
        t3 = time.perf_counter()

        if self.transform:
            img = self.transform(img)

        # Store timing (accessible by DataLoader)
        self._last_io_time = t1 - t0
        self._last_decode_time = t2 - t1
        self._last_preprocess_time = t3 - t2
        self._last_total_time = t3 - t0

        return img, label

    def preprocess(self, img: Image.Image, label):
        img = img.resize((64, 64))
        return T.ToTensor()(img), label
