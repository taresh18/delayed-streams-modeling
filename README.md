# delayed-streams-modeling
Delayed Streams Modeling (DSM) is a flexible formulation for streaming, multimodal sequence-to-sequence learning.

## Speech To Text

### PyTorch implementation

```bash
python -m moshi.run_inference --hf-repo kyutai/stt input.mp3
```

### MLX implementation

```bash
python -m moshi_mlx.run_inference --hf-repo kyutai/stt-mlx ~/tmp/bria-24khz.mp3 --temp 0
```

## License

The present code is provided under the MIT license for the Python parts, and Apache license for the Rust backend.
The web client code is provided under the MIT license.
Note that parts of this code is based on [AudioCraft](https://github.com/facebookresearch/audiocraft), released under
the MIT license.

The weights for the models are released under the CC-BY 4.0 license.
