from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset


class H5SatelliteDataset(Dataset):
    def __init__(
        self,
        database_h5: Path,
        resize: tuple[int, int],
        map_path: Path | None = None,
        crop_size_px: int | None = None,
    ) -> None:
        self.database_h5 = database_h5
        self.resize = resize
        self.map_path = map_path
        self.crop_size_px = crop_size_px
        with h5py.File(database_h5, "r") as h5:
            self.has_image_data = "image_data" in h5
            self.names = [name.decode("utf-8") for name in h5["image_name"][:]]
            if self.crop_size_px is None:
                self.crop_size_px = int(h5.attrs.get("tile_size_px", 1536))
            if self.map_path is None and "map_path" in h5.attrs:
                self.map_path = Path(h5.attrs["map_path"])
        if not self.has_image_data and self.map_path is None:
            raise ValueError(f"{database_h5} has no image_data; pass --map_path or store map_path attr.")
        self.h5 = None
        self.map_image = None
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        if self.has_image_data:
            if self.h5 is None:
                self.h5 = h5py.File(self.database_h5, "r", swmr=True)
            image = Image.fromarray(self.h5["image_data"][index]).convert("RGB")
        else:
            if self.map_image is None:
                Image.MAX_IMAGE_PIXELS = None
                self.map_image = Image.open(self.map_path).convert("RGB")
            name = self.names[index]
            y = int(round(float(name.split("@")[1])))
            x = int(round(float(name.split("@")[2])))
            half = self.crop_size_px // 2
            image = self.map_image.crop((x - half, y - half, x + half, y + half)).convert("RGB")
        return self.transform(image), self.names[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate STHN query H5 images from satellite database chips using the paper TGM Pix2Pix generator."
    )
    parser.add_argument("--database_h5", required=True)
    parser.add_argument("--output_queries_h5", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map_path", default=None)
    parser.add_argument("--crop_size_px", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resize", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--G_net", default="unet", choices=["unet", "unet_deep"])
    parser.add_argument("--GAN_norm", default="batch", choices=["batch", "instance"])
    parser.add_argument("--GAN_upsample", default="bilinear", choices=["bilinear", "convtrans"])
    parser.add_argument("--G_tanh", action="store_true")
    return parser.parse_args()


def load_tgm(repo_root: Path, args: argparse.Namespace):
    sys.path.insert(0, str(repo_root / "global_pipeline"))
    from model.pix2pix_networks.networks import UnetGenerator

    if args.G_net == "unet":
        net_g = UnetGenerator(
            3, 1, 8, norm=args.GAN_norm, upsample=args.GAN_upsample, use_tanh=args.G_tanh
        )
    elif args.G_net == "unet_deep":
        net_g = UnetGenerator(
            3, 1, 9, norm=args.GAN_norm, upsample=args.GAN_upsample, use_tanh=args.G_tanh
        )
    else:
        raise NotImplementedError(args.G_net)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if "model_netG_state_dict" not in checkpoint:
        raise KeyError(f"{args.checkpoint} does not contain model_netG_state_dict")
    state_dict = checkpoint["model_netG_state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    net_g.load_state_dict(state_dict)
    net_g = net_g.to(args.device).eval()
    return SimpleNamespace(netG=net_g)


def generated_to_uint8_rgb(batch: torch.Tensor) -> np.ndarray:
    batch = torch.clamp(batch, min=-1, max=1)
    batch = batch * 0.5 + 0.5
    arrays = []
    to_pil = transforms.ToPILImage()
    for image in batch.cpu():
        pil = to_pil(image)
        pil = ImageOps.grayscale(pil).convert("RGB")
        arrays.append(np.asarray(pil, dtype=np.uint8))
    return np.stack(arrays, axis=0)


def write_queries(args: argparse.Namespace, model) -> None:
    database_h5 = Path(args.database_h5)
    output_h5 = Path(args.output_queries_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    if output_h5.exists():
        output_h5.unlink()

    dataset = H5SatelliteDataset(
        database_h5,
        resize=tuple(args.resize),
        map_path=Path(args.map_path) if args.map_path else None,
        crop_size_px=args.crop_size_px,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    names: list[str] = []
    image_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for satellite, batch_names in loader:
            satellite = satellite.to(args.device)
            model.netG = model.netG.to(args.device)
            generated = model.netG(satellite)
            image_chunks.append(generated_to_uint8_rgb(generated))
            names.extend(batch_names)

    images = np.concatenate(image_chunks, axis=0)
    sizes = np.asarray([[args.resize[0], args.resize[1]]] * len(names), dtype=np.int32)
    with h5py.File(output_h5, "w") as h5:
        h5.attrs["source"] = "paper_tgm_pix2pix"
        h5.attrs["database_h5"] = str(database_h5.resolve())
        h5.attrs["checkpoint"] = str(Path(args.checkpoint).resolve())
        h5.create_dataset("image_data", data=images, chunks=(1, args.resize[0], args.resize[1], 3), compression="lzf")
        h5.create_dataset("image_size", data=sizes, compression="lzf")
        h5.create_dataset("image_name", data=names, dtype=h5py.string_dtype(encoding="utf-8"), compression="lzf")
    print(f"Wrote {len(names)} TGM query images to {output_h5}")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    repo_root = Path(__file__).resolve().parents[1]
    model = load_tgm(repo_root, args)
    write_queries(args, model)


if __name__ == "__main__":
    main()
