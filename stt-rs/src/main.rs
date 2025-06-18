// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use candle::{Device, Tensor};
use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    /// The audio input file, in wav/mp3/ogg/... format.
    in_file: String,

    /// The repo where to get the model from.
    #[arg(long, default_value = "kyutai/stt-1b-en_fr-candle")]
    hf_repo: String,

    /// Run the model on cpu.
    #[arg(long)]
    cpu: bool,
}

fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

#[derive(Debug, serde::Deserialize)]
struct Config {
    mimi_name: String,
    tokenizer_name: String,
    card: usize,
    text_card: usize,
    dim: usize,
    n_q: usize,
    context: usize,
    max_period: f64,
    num_heads: usize,
    num_layers: usize,
    causal: bool,
}

impl Config {
    fn model_config(&self) -> moshi::lm::Config {
        let lm_cfg = moshi::transformer::Config {
            d_model: self.dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            dim_feedforward: self.dim * 4,
            causal: self.causal,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: self.context,
            max_period: self.max_period as usize,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: moshi::NormType::RmsNorm,
            positional_embedding: moshi::transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096 * 4,
            shared_cross_attn: false,
        };
        moshi::lm::Config {
            transformer: lm_cfg,
            depformer: None,
            audio_vocab_size: self.card + 1,
            text_in_vocab_size: self.text_card + 1,
            text_out_vocab_size: self.text_card,
            audio_codebooks: self.n_q,
            conditioners: Default::default(),
            extra_heads: None,
        }
    }
}

struct Model {
    state: moshi::asr::State,
    text_tokenizer: sentencepiece::SentencePieceProcessor,
    dev: Device,
}

impl Model {
    fn load_from_hf(hf_repo: &str, dev: &Device) -> Result<Self> {
        let dtype = dev.bf16_default_to_f32();

        // Retrieve the model files from the Hugging Face Hub
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(hf_repo.to_string());
        let config_file = repo.get("config.json")?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
        let tokenizer_file = repo.get(&config.tokenizer_name)?;
        let model_file = repo.get("model.safetensors")?;
        let mimi_file = repo.get(&config.mimi_name)?;

        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&tokenizer_file)?;
        let vb_lm =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[&model_file], dtype, dev)? };
        let audio_tokenizer = moshi::mimi::load(mimi_file.to_str().unwrap(), Some(32), dev)?;
        let lm = moshi::lm::LmModel::new(
            &config.model_config(),
            moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm),
        )?;
        let state = moshi::asr::State::new(1, 0, 0., audio_tokenizer, lm)?;
        Ok(Model {
            state,
            text_tokenizer,
            dev: dev.clone(),
        })
    }

    fn run(&mut self, pcm: &[f32]) -> Result<()> {
        use std::io::Write;

        for pcm in pcm.chunks(1920) {
            let pcm = Tensor::new(pcm, &self.dev)?.reshape((1, 1, ()))?;
            let asr_msgs = self.state.step_pcm(pcm, None, &().into(), |_, _, _| ())?;
            let mut prev_text_token = 0;
            for asr_msg in asr_msgs.iter() {
                match asr_msg {
                    moshi::asr::AsrMsg::Step { .. } | moshi::asr::AsrMsg::EndWord { .. } => {}
                    moshi::asr::AsrMsg::Word { tokens, .. } => {
                        for &text_token in tokens.iter() {
                            let s = {
                                let prev_ids =
                                    self.text_tokenizer.decode_piece_ids(&[prev_text_token]);
                                let ids = self
                                    .text_tokenizer
                                    .decode_piece_ids(&[prev_text_token, text_token]);
                                prev_text_token = text_token;
                                prev_ids.and_then(|prev_ids| {
                                    ids.map(|ids| {
                                        if ids.len() > prev_ids.len() {
                                            ids[prev_ids.len()..].to_string()
                                        } else {
                                            String::new()
                                        }
                                    })
                                })?
                            };
                            print!("{s}");
                            std::io::stdout().flush()?
                        }
                    }
                }
            }
        }
        println!();
        Ok(())
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = device(args.cpu)?;
    println!("Using device: {:?}", device);

    println!("Loading audio file from: {}", args.in_file);
    let (pcm, sample_rate) = kaudio::pcm_decode(args.in_file)?;
    let mut pcm = if sample_rate != 24_000 {
        kaudio::resample(&pcm, sample_rate as usize, 24_000)?
    } else {
        pcm
    };
    // Add some silence at the end to ensure all the audio is processed.
    pcm.resize(pcm.len() + 1920 * 32, 0.0);
    println!("Loading model from repository: {}", args.hf_repo);
    let mut model = Model::load_from_hf(&args.hf_repo, &device)?;
    println!("Running inference");
    model.run(&pcm)?;
    Ok(())
}
