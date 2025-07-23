## SESA-KAN

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>


This is a implementation of the paper [SESA-KAN: Simultaneous estimation of sex and age from dental panoramic radiographs](https://arxiv.org/abs/2111.06377):
```
@Article{Liang2025,
  author  = {Julian Liang, Igor Borja, David Lima, Márcio Júnior, Patrícia Cury, Luciano Oliveira},
  journal = {arXiv:2111.06377},
  title   = {SESA-KAN: Simultaneous estimation of sex and age from dental panoramic radiographs},
  year    = {2025},
}
```

* The original implementation was in TensorFlow+TPU. This re-implementation is in PyTorch+GPU.

### Catalog

- [x] Visualization demo
- [x] Pre-trained checkpoints + fine-tuning code
- [x] Pre-training code

### Visualization demo

Run our interactive visualization demo using [Colab notebook](https://colab.research.google.com/github/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb) (no GPU needed):
<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/147859292-77341c70-2ed8-4703-b153-f505dcb6f2f8.png" width="600">
</p>

### Fine-tuning with pre-trained checkpoints

The following table provides the pre-trained checkpoints used in the paper, converted from TF/TPU to PT/GPU:
<table><tbody>

The fine-tuning instruction is in [FINETUNE.md](FINETUNE.md).

By fine-tuning these pre-trained models, we rank #1 in these classification tasks (detailed in the paper):
<table><tbody>


### Pre-training

The pre-training instruction is in [PRETRAIN.md](PRETRAIN.md).

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
