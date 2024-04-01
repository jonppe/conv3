#!/bin/env python
from lightning.pytorch.cli import LightningCLI

from conv3.train import Conv3Module, BookDataModule


def cli_main():
    _ = LightningCLI(Conv3Module, BookDataModule)


if __name__ == "__main__":
    cli_main()
