# Learn CLIP

Build image

```bash
docker build -t ghcr.io/yanting-yang/fmri_clip:250123 .
```

Test image

```bash
docker run -it --rm --gpus=all --ipc=host ghcr.io/yanting-yang/fmri_clip:250123
```

Push image

```bash
docker push ghcr.io/yanting-yang/fmri_clip:250123
```
