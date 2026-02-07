"""
Patch for ACE-Step SFT model: fix repeated CFG doubling in non-cover path.

The upstream modeling_acestep_v15_base.py has a bug in generate_audio() where
context_latents_non_cover gets doubled on EVERY diffusion step after cover_steps,
instead of just once. This causes a tensor dimension mismatch crash.

Apply this patch after model download:
    python patches/sft_cfg_fix.py

Or it will be applied automatically by the handler if detected.
"""
import os
import sys

def patch_sft_model():
    """Find and patch the SFT model's generate_audio method."""
    import glob
    
    # Search HF cache for the SFT model file
    patterns = [
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/acestep_hyphen_v15_hyphen_sft/modeling_acestep_v15_base.py"),
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/*/modeling_acestep_v15_base.py"),
    ]
    
    target_files = []
    for pattern in patterns:
        target_files.extend(glob.glob(pattern))
    
    if not target_files:
        print("SFT model file not found in HF cache. Download the model first.")
        return False
    
    patched = 0
    for filepath in target_files:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if '_non_cover_cfg_applied' in content:
            print(f"Already patched: {filepath}")
            continue
        
        # The bug: CFG doubling inside the loop without a guard
        old = '''                if step_idx >= cover_steps:
                    if do_cfg_guidance:
                        encoder_hidden_states_non_cover = torch.cat([encoder_hidden_states_non_cover, self.null_condition_emb.expand_as(encoder_hidden_states_non_cover)], dim=0)
                        encoder_attention_mask_non_cover = torch.cat([encoder_attention_mask_non_cover, encoder_attention_mask_non_cover], dim=0)
                        # src_latents
                        context_latents_non_cover = torch.cat([context_latents_non_cover, context_latents_non_cover], dim=0)

                    encoder_hidden_states = encoder_hidden_states_non_cover'''
        
        new = '''                if step_idx >= cover_steps:
                    if do_cfg_guidance and not _non_cover_cfg_applied:
                        encoder_hidden_states_non_cover = torch.cat([encoder_hidden_states_non_cover, self.null_condition_emb.expand_as(encoder_hidden_states_non_cover)], dim=0)
                        encoder_attention_mask_non_cover = torch.cat([encoder_attention_mask_non_cover, encoder_attention_mask_non_cover], dim=0)
                        # src_latents
                        context_latents_non_cover = torch.cat([context_latents_non_cover, context_latents_non_cover], dim=0)
                        _non_cover_cfg_applied = True

                    encoder_hidden_states = encoder_hidden_states_non_cover'''
        
        if old not in content:
            print(f"Pattern not found (may be different version): {filepath}")
            continue
        
        # Also need to add the flag initialization before the loop
        loop_marker = "        with torch.no_grad():\n            for step_idx, (t_curr, t_prev) in enumerate(iterator):"
        if loop_marker in content:
            content = content.replace(loop_marker, "        _non_cover_cfg_applied = False\n" + loop_marker)
        
        content = content.replace(old, new)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"Patched: {filepath}")
        patched += 1
    
    return patched > 0

if __name__ == "__main__":
    success = patch_sft_model()
    sys.exit(0 if success else 1)
