Small experiment on using Conv3D for language modelling.
Based on an example in
- https://twitter.com/canalCCore2/status/1773490328110047477

With some more description in
- https://twitter.com/canalCCore2/status/1773142723559567488
- https://twitter.com/canalCCore2/status/1773992830840496629

## Usage

Get the simplebooks dataset by running setup.sh.

```
./setup.sh [optional/dest/dir]
```

The model can be trained on command line using standard pytorch lightning commands:

for example

```
./cli.py fit
./cli.py fit --help
cli.py fit --ckpt_path ./lightning_logs/version_17/checkpoints/epoch\=254-step\=255.ckpt
```

## Development

There's .devcontainer spec that allows easy development especially with VSCode (it might work with e.g. PyCharm).
The version has been written for Podman+Cuda combination. Using Cuda with Docker might require small changes.
