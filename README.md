# delayed-streams-modeling
Delayed Streams Modeling (DSM) is a flexible formulation for streaming, multimodal sequence-to-sequence learning.

## Speech To Text

### Leaderboard model
The leaderboard model handles english only, it has ~2.6B parameters.

#### PyTorch implementation
[[Hugging Face]](https://huggingface.co/kyutai/stt)

```bash
# wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
python -m moshi.run_inference --hf-repo kyutai/stt bria.mp3
```

#### MLX implementation
[[Hugging Face]](https://huggingface.co/kyutai/stt-mlx)

```bash
# wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
python -m moshi_mlx.run_inference --hf-repo kyutai/stt-mlx bria.mp3 --temp 0
```

#### Rust implementation
[[Hugging Face]](https://huggingface.co/kyutai/stt-candle)

The Rust implementation provides a server that can process multiple streaming
queries in parallel. Dependening on the amount of memory on your GPU, you may
have to adjust the batch size from the config file. For a L40S GPU, a batch size
of 64 works well.

In order to run the server, install the `moshi-server` crate via the following
command. The server code can be found in the
[kyutai-labs/moshi](https://github.com/kyutai-labs/moshi/tree/main/rust/moshi-server)
repository.
```bash
cargo install --features cuda moshi-server
```

Then the server can be started via the following command using the config file
from this repository.
```bash
moshi-server worker --config configs/config-stt-hf.toml
```

Once the server has started you can run a streaming inference with the following
script.
```bash
# wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
uv run scripts/asr-streaming-query.py bria.mp3
```

The script simulates some real-time processing of the audio. Faster processing
can be triggered by setting the real-time factor, e.g. `--rtf 500` will process
the data as fast as possible.

## Text To Speech

We're in the process of open-sourcing our TTS models. Check back for updates!

## License

The present code is provided under the MIT license for the Python parts, and Apache license for the Rust backend.
The web client code is provided under the MIT license.
Note that parts of this code is based on [AudioCraft](https://github.com/facebookresearch/audiocraft), released under
the MIT license.

The weights for the models are released under the CC-BY 4.0 license.
