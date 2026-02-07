"""
MLX-LM backend for ACE-Step 5Hz LM inference on Apple Silicon.

Provides fast, Metal-accelerated LM inference as an alternative to nano-vllm (CUDA)
and the PyTorch backend. Uses mlx-lm for optimized transformer inference with KV
caching, quantization support, and native Metal acceleration.

Key features:
- Loads Qwen3-based 5Hz LM models directly from HuggingFace format (safetensors)
- Supports 0.6B and 1.7B model sizes
- Supports 4-bit and 8-bit quantization for memory efficiency
- Integrates with ACE-Step's constrained decoding via logits processor adapter
- Streaming generation for progress updates
- Clean memory management via MLX's metal memory pool

Constrained Decoding Approach:
    mlx-lm supports logits_processors with signature (tokens: mx.array, logits: mx.array) -> mx.array.
    We wrap ACE-Step's existing MetadataConstrainedLogitsProcessor by converting between mx.array
    and torch.Tensor at the boundary. This reuses the battle-tested FSM-based constrained
    decoding logic without reimplementing it in MLX. The conversion overhead is negligible
    compared to the model forward pass.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from loguru import logger

# ── Availability sentinel ────────────────────────────────────────────────────
MLX_LM_AVAILABLE = False
_MLX_IMPORT_ERROR: Optional[str] = None

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as mlx_load
    from mlx_lm import stream_generate as mlx_stream_generate
    from mlx_lm import generate as mlx_generate
    from mlx_lm.generate import generate_step, GenerationResponse
    from mlx_lm.sample_utils import make_sampler

    MLX_LM_AVAILABLE = True
except ImportError as e:
    _MLX_IMPORT_ERROR = str(e)
except Exception as e:
    _MLX_IMPORT_ERROR = f"Unexpected error importing mlx-lm: {e}"


def is_available() -> bool:
    """Check if the MLX-LM backend is available."""
    return MLX_LM_AVAILABLE


def get_import_error() -> Optional[str]:
    """Return the import error message, if any."""
    return _MLX_IMPORT_ERROR


# ── Constrained Decoding Adapter ─────────────────────────────────────────────

class MLXConstrainedLogitsAdapter:
    """
    Adapter that wraps ACE-Step's MetadataConstrainedLogitsProcessor for use
    with mlx-lm's logits_processors interface.

    mlx-lm logits_processors signature:
        (tokens: mx.array, logits: mx.array) -> mx.array
        - tokens: 1D array of all tokens generated so far
        - logits: 1D or 2D array of shape [vocab_size] or [1, vocab_size]

    ACE-Step's MetadataConstrainedLogitsProcessor.__call__ signature:
        (input_ids: torch.Tensor[batch, seq], scores: torch.Tensor[batch, vocab]) -> torch.Tensor
        + update_state(token_id: int) called after each token is sampled

    This adapter bridges the gap by:
    1. Converting mx.array → torch.Tensor for the processor call
    2. Converting torch.Tensor → mx.array for the return value
    3. Calling update_state() with each generated token
    """

    def __init__(self, constrained_processor):
        """
        Args:
            constrained_processor: An instance of MetadataConstrainedLogitsProcessor
        """
        self.processor = constrained_processor
        self._call_count = 0  # Number of times __call__ has been invoked
        self._prev_token_count = 0  # Track tokens to detect newly generated ones

    def __call__(self, tokens: "mx.array", logits: "mx.array") -> "mx.array":
        """
        Apply constrained decoding to logits.

        mlx-lm calls logits_processors with:
          - tokens: all tokens so far (prompt + generated)
          - logits: [vocab_size] logits for the next token to sample

        Lifecycle per generation step:
          1. Model forward pass → raw logits
          2. This processor called (modify logits based on FSM state)
          3. Sampler picks a token from modified logits
          4. Token appended to sequence
          5. Next call: tokens now includes the newly sampled token

        So on call N (N >= 1), tokens[-1] is the token sampled in step N-1.
        We call update_state(token) to advance the FSM BEFORE masking for step N.

        On the first call (N=0), tokens is just the prompt — no update_state needed.

        Args:
            tokens: 1D mx.array of token IDs (prompt + generated so far)
            logits: 1D mx.array of shape [vocab_size]

        Returns:
            Modified logits as mx.array
        """
        import torch

        tokens_list = tokens.tolist() if tokens.ndim > 0 else [tokens.item()]
        current_count = len(tokens_list)

        # Update FSM state for any newly generated token(s) since last call
        # Skip on the very first call (tokens are just the prompt)
        if self._call_count > 0 and current_count > self._prev_token_count:
            # Typically only 1 new token, but handle multiple for safety
            for idx in range(self._prev_token_count, current_count):
                self.processor.update_state(tokens_list[idx])

        self._prev_token_count = current_count
        self._call_count += 1

        # Convert to torch tensors for the processor
        input_ids = torch.tensor([tokens_list], dtype=torch.long)  # [1, seq_len]
        logits_list = logits.tolist() if logits.ndim > 0 else [logits.item()]
        scores = torch.tensor([logits_list], dtype=torch.float32)  # [1, vocab_size]

        # Apply constrained decoding (FSM-based masking)
        modified_scores = self.processor(input_ids, scores)

        # Convert back to mx.array
        return mx.array(modified_scores[0].tolist())

    def reset(self, prompt_length: int = 0):
        """Reset adapter state for a new generation."""
        self._call_count = 0
        self._prev_token_count = prompt_length


# ── MLX-LM Backend ───────────────────────────────────────────────────────────

class MLXLMBackend:
    """
    MLX-LM backend for ACE-Step 5Hz LM inference on Apple Silicon.

    Provides a high-level interface matching what llm_inference.py expects,
    using mlx-lm for model loading and generation.
    """

    def __init__(self):
        if not MLX_LM_AVAILABLE:
            raise RuntimeError(
                f"mlx-lm is not available: {_MLX_IMPORT_ERROR}. "
                "Install with: pip install mlx-lm"
            )
        self.model = None
        self.tokenizer = None
        self.model_path: Optional[str] = None
        self._config: Optional[Dict[str, Any]] = None

    # ── Model Loading ─────────────────────────────────────────────────────

    def load_model(
        self,
        model_path: str,
        quantize: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Load a Qwen3-based 5Hz LM model using mlx-lm.

        mlx-lm can load HuggingFace models directly from a local directory
        containing config.json + model*.safetensors files — no conversion
        needed for supported architectures (Qwen3 is natively supported).

        Args:
            model_path: Path to model directory (HuggingFace format)
            quantize: Optional quantization bits (4 or 8). None = no quantization.
                      Quantization is applied on-the-fly by mlx-lm and saves memory.

        Returns:
            (success: bool, status_message: str)
        """
        try:
            start_time = time.time()
            logger.info(f"Loading MLX-LM model from {model_path}")

            # Build model_config for quantization if requested
            model_config = None
            if quantize is not None:
                if quantize not in (4, 8):
                    return False, f"❌ Invalid quantization bits: {quantize}. Must be 4 or 8."
                logger.info(f"Applying {quantize}-bit quantization")
                # mlx-lm handles quantization via the model weights format
                # For on-the-fly quantization, we'd need to convert first.
                # If the model is already in MLX quantized format, it loads directly.
                # For HF safetensors, we load full precision and can quantize after.

            self.model, self.tokenizer, self._config = mlx_load(
                model_path,
                return_config=True,
            )
            self.model_path = model_path

            load_time = time.time() - start_time

            # Apply post-load quantization if requested and model isn't already quantized
            if quantize is not None and not self._is_quantized():
                logger.info(f"Applying {quantize}-bit post-load quantization...")
                quant_start = time.time()
                nn.quantize(self.model, bits=quantize)
                mx.eval(self.model.parameters())
                quant_time = time.time() - quant_start
                logger.info(f"Quantization applied in {quant_time:.2f}s")

            # Get model info for status message
            model_type = self._config.get("model_type", "unknown")
            vocab_size = self._config.get("vocab_size", "?")
            num_layers = self._config.get("num_hidden_layers", "?")
            hidden_size = self._config.get("hidden_size", "?")

            status = (
                f"✅ 5Hz LM initialized successfully (MLX-LM)\n"
                f"Model: {model_path}\n"
                f"Type: {model_type} | Layers: {num_layers} | Hidden: {hidden_size} | Vocab: {vocab_size}\n"
                f"Backend: MLX (Metal-accelerated)\n"
                f"Load time: {load_time:.2f}s"
            )
            if quantize is not None:
                status += f"\nQuantization: {quantize}-bit"

            logger.info(f"MLX-LM model loaded successfully in {load_time:.2f}s "
                        f"(type={model_type}, layers={num_layers}, hidden={hidden_size})")
            return True, status

        except FileNotFoundError as e:
            msg = f"❌ Model files not found at {model_path}: {e}"
            logger.error(msg)
            return False, msg
        except Exception as e:
            import traceback
            msg = f"❌ Error loading MLX-LM model: {e}\n\n{traceback.format_exc()}"
            logger.error(msg)
            return False, msg

    def _is_quantized(self) -> bool:
        """Check if the loaded model is already quantized."""
        if self._config is None:
            return False
        return "quantization" in self._config

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None and self.tokenizer is not None

    # ── Generation ────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.85,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        logits_processor: Optional[Callable] = None,
        on_token: Optional[Callable[[GenerationResponse], bool]] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt string (already formatted with chat template)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold (None = disabled)
            top_k: Top-K sampling (None or 0 = disabled)
            repetition_penalty: Repetition penalty (1.0 = none). NOTE: mlx-lm
                does not natively support repetition penalty in its sampler.
                When != 1.0, we apply it via a logits processor.
            logits_processor: Optional MLX-compatible logits processor function
                with signature (tokens: mx.array, logits: mx.array) -> mx.array
            on_token: Optional callback called for each generated token.
                Receives a GenerationResponse. Return True to stop generation early.

        Returns:
            Generated text string (excluding the prompt)
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Build sampler
        sampler = make_sampler(
            temp=temperature,
            top_p=top_p if top_p is not None and 0 < top_p < 1.0 else 0.0,
            top_k=top_k if top_k is not None and top_k > 0 else 0,
        )

        # Build logits processors list
        processors = []
        if repetition_penalty != 1.0:
            processors.append(
                _make_repetition_penalty_processor(repetition_penalty)
            )
        if logits_processor is not None:
            processors.append(logits_processor)

        # Use stream_generate for token-by-token output (enables early stopping)
        # response.text is an incremental segment, concatenate to build full output
        generated_text = ""
        for response in mlx_stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=processors if processors else None,
        ):
            generated_text += response.text
            if on_token is not None:
                if on_token(response):
                    break

        return generated_text

    def generate_with_cfg(
        self,
        prompt: str,
        unconditional_prompt: str,
        cfg_scale: float = 2.0,
        max_tokens: int = 4096,
        temperature: float = 0.85,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        logits_processor: Optional[Callable] = None,
        on_token: Optional[Callable[[int, str], bool]] = None,
    ) -> str:
        """
        Generate text with Classifier-Free Guidance (CFG).

        CFG formula: logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)

        Since mlx-lm doesn't natively support CFG, we implement a custom generation
        loop that runs both conditional and unconditional forward passes and combines
        the logits.

        Args:
            prompt: Conditional prompt (formatted)
            unconditional_prompt: Unconditional prompt for CFG
            cfg_scale: CFG scale factor (1.0 = no CFG)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-K sampling
            repetition_penalty: Repetition penalty
            logits_processor: Optional MLX logits processor
            on_token: Optional callback (token_id, text_so_far) -> should_stop

        Returns:
            Generated text string
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Call load_model() first.")

        model = self.model

        # Encode both prompts
        cond_tokens = self._encode_prompt(prompt)
        uncond_tokens = self._encode_prompt(unconditional_prompt)

        cond_prompt = mx.array(cond_tokens)
        uncond_prompt = mx.array(uncond_tokens)

        # Build sampler for final token selection
        sampler = make_sampler(
            temp=temperature,
            top_p=top_p if top_p is not None and 0 < top_p < 1.0 else 0.0,
            top_k=top_k if top_k is not None and top_k > 0 else 0,
        )

        # Run conditional prefill
        cond_cache = None
        uncond_cache = None

        # Run generation loop
        generated_tokens = []
        all_cond_tokens = cond_tokens.copy()

        # Prefill both caches
        cond_logits = self._prefill(model, cond_prompt)
        uncond_logits = self._prefill(model, uncond_prompt, cache_name="_uncond_cache")
        cond_cache = getattr(self, "_cond_cache", None)
        uncond_cache = getattr(self, "_uncond_cache", None)

        # Process up to max_tokens
        eos_ids = set()
        if hasattr(self.tokenizer, 'eos_token_ids'):
            eos_ids = set(self.tokenizer.eos_token_ids)
        elif hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            eos_ids = {self.tokenizer.eos_token_id}

        # Initialize detokenizer
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        for step in range(max_tokens):
            # Apply CFG formula
            cfg_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)

            # Apply logits processors
            if logits_processor is not None:
                tokens_so_far = mx.array(all_cond_tokens)
                cfg_logits = logits_processor(tokens_so_far, cfg_logits)

            if repetition_penalty != 1.0:
                rp_proc = _make_repetition_penalty_processor(repetition_penalty)
                tokens_so_far = mx.array(all_cond_tokens)
                cfg_logits = rp_proc(tokens_so_far, cfg_logits)

            # Sample token
            token = sampler(cfg_logits)
            token_id = token.item()

            # Check EOS
            if token_id in eos_ids:
                break

            generated_tokens.append(token_id)
            all_cond_tokens.append(token_id)

            # Update detokenizer
            detokenizer.add_token(token_id)

            # Callback
            if on_token is not None:
                if on_token(token_id, detokenizer.text):
                    break

            # Forward pass with the new token for both streams
            token_mx = mx.array([[token_id]])
            cond_logits = self._step(model, token_mx, cache_name="_cond_cache")
            uncond_logits = self._step(model, token_mx, cache_name="_uncond_cache")

        # Decode all generated tokens
        detokenizer.finalize()
        return detokenizer.text

    def _encode_prompt(self, prompt: str) -> List[int]:
        """Encode a prompt string to token IDs."""
        add_special_tokens = (
            self.tokenizer.bos_token is None
            or not prompt.startswith(self.tokenizer.bos_token)
        )
        return self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

    def _prefill(self, model, prompt: "mx.array", cache_name: str = "_cond_cache"):
        """Run prefill on prompt and cache KV state. Returns last-position logits [vocab]."""
        cache = None
        # Create new KV cache
        if hasattr(model, "make_cache"):
            cache = model.make_cache()
        else:
            # Fallback: create cache manually for each layer
            cache = [None] * (getattr(model, "n_layers", 0) or len(getattr(model, "layers", [])))

        setattr(self, cache_name, cache)

        # Run prefill in chunks for memory efficiency
        step_size = 2048
        n_tokens = prompt.shape[0]

        for start in range(0, n_tokens, step_size):
            end = min(start + step_size, n_tokens)
            chunk = prompt[start:end]
            logits = model(chunk[None, :], cache=cache)
            mx.eval([c.state for c in cache if hasattr(c, "state")])

        # Return logits for last position
        return logits[0, -1, :]  # [vocab_size]

    def _step(self, model, token: "mx.array", cache_name: str = "_cond_cache"):
        """Run one step of generation. Returns logits for the new token [vocab]."""
        cache = getattr(self, cache_name, None)
        logits = model(token, cache=cache)
        mx.eval([c.state for c in cache if hasattr(c, "state")])
        return logits[0, -1, :]

    # ── Memory Management ─────────────────────────────────────────────────

    def clear_memory(self):
        """Release model and clear MLX memory pool."""
        self.model = None
        self.tokenizer = None
        self._config = None
        self.model_path = None
        # Clean up any cached KV states
        if hasattr(self, "_cond_cache"):
            del self._cond_cache
        if hasattr(self, "_uncond_cache"):
            del self._uncond_cache

        if MLX_LM_AVAILABLE:
            mx.metal.clear_cache()
            logger.info("MLX memory cache cleared")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current MLX Metal memory usage in GB."""
        if not MLX_LM_AVAILABLE:
            return {}
        try:
            active = mx.metal.get_active_memory() / (1024 ** 3)
            peak = mx.metal.get_peak_memory() / (1024 ** 3)
            cache = mx.metal.get_cache_memory() / (1024 ** 3)
            return {
                "active_gb": round(active, 2),
                "peak_gb": round(peak, 2),
                "cache_gb": round(cache, 2),
            }
        except Exception:
            return {}


# ── Utility Functions ─────────────────────────────────────────────────────────

def _make_repetition_penalty_processor(
    penalty: float,
) -> Callable:
    """
    Create a repetition penalty logits processor for mlx-lm.

    Args:
        penalty: Repetition penalty factor (>1.0 penalizes repetition)

    Returns:
        A function with signature (tokens: mx.array, logits: mx.array) -> mx.array
    """
    def processor(tokens: "mx.array", logits: "mx.array") -> "mx.array":
        if tokens.size == 0:
            return logits

        # Get unique tokens that have been generated
        unique_tokens = set(tokens.tolist())

        # Apply penalty: divide positive logits, multiply negative logits
        logits_list = logits.tolist()
        for token_id in unique_tokens:
            if 0 <= token_id < len(logits_list):
                if logits_list[token_id] > 0:
                    logits_list[token_id] /= penalty
                else:
                    logits_list[token_id] *= penalty

        return mx.array(logits_list)

    return processor
