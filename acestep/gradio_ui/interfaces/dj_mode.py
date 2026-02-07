"""
DJ Mode Gradio Interface

Provides a Gradio tab for the AI DJ feature — LLM-powered setlist planning
and sequential music generation via ACE-Step.
"""

import gradio as gr
import json
import time
from typing import Optional, Dict, Any

from acestep.gradio_ui.i18n import t
from acestep.dj_mode import (
    AVAILABLE_GENRES,
    LLMProvider,
    MoodProgression,
    TransitionStyle,
    DEFAULT_MODELS,
    PROVIDER_MODELS as DJ_PROVIDER_MODELS,
    DJConfig,
    DJEngine,
    plan_setlist,
)


# Use the model lists from dj_mode.py (single source of truth)
PROVIDER_MODELS = {
    "ollama": DJ_PROVIDER_MODELS[LLMProvider.OLLAMA],
    "openrouter": DJ_PROVIDER_MODELS[LLMProvider.OPENROUTER],
    "gemini": DJ_PROVIDER_MODELS[LLMProvider.GEMINI],
}


def _update_model_choices(provider: str):
    """Return model dropdown update when provider changes."""
    models = PROVIDER_MODELS.get(provider, [])
    default = models[0] if models else ""
    return gr.update(choices=models, value=default)


def _needs_api_key(provider: str):
    """Return whether API key field should be visible."""
    return gr.update(visible=(provider != "ollama"))


def _format_setlist_display(plan_data: Dict[str, Any]) -> str:
    """Format a setlist plan into a readable markdown string."""
    if not plan_data:
        return ""
    lines = [f"### 🎵 {plan_data.get('narrative', 'DJ Set')}\n"]
    for track in plan_data.get("tracks", []):
        energy_bar = "▓" * int(track["energy"] * 10) + "░" * (10 - int(track["energy"] * 10))
        lines.append(
            f"**{track['index'] + 1}. {track['title']}** — {track['genre']}\n"
            f"  🎹 {track['key']} | ♩ {track['bpm']} BPM | "
            f"⚡ {energy_bar} ({track['energy']:.1f})\n"
            f"  *{track['caption'][:120]}{'...' if len(track.get('caption', '')) > 120 else ''}*\n"
        )
    return "\n".join(lines)


def create_dj_mode_section(dit_handler, llm_handler, init_params=None) -> dict:
    """Create the DJ Mode tab for the Gradio UI.

    Args:
        dit_handler: AceStepHandler instance.
        llm_handler: LLMHandler instance.
        init_params: Initialization parameters (for checking service mode, etc.)

    Returns:
        Dictionary of Gradio components for event handler wiring.
    """
    # Hide DJ mode in service mode (it's experimental)
    service_mode = init_params and init_params.get("service_mode", False)

    with gr.Accordion("🎧 AI DJ Mode (Experimental)", open=False, visible=not service_mode):
        gr.Markdown(
            "Let an AI DJ plan and generate a complete music set based on your vibe description. "
            "Powered by an LLM for creative setlist planning + ACE-Step for music generation."
        )

        with gr.Row():
            # ---- Left column: Configuration ----
            with gr.Column(scale=1):
                vibe_input = gr.Textbox(
                    label="Vibe / Mood Description",
                    placeholder="e.g., 'chill evening coffee shop vibes transitioning into an upbeat dance party'",
                    lines=3,
                    max_lines=6,
                )

                with gr.Group():
                    gr.Markdown("#### LLM Settings")
                    provider_dropdown = gr.Dropdown(
                        choices=["ollama", "openrouter", "gemini"],
                        value="openrouter",
                        label="LLM Provider",
                        info="Ollama = free & local, others need API keys",
                    )
                    model_dropdown = gr.Dropdown(
                        choices=PROVIDER_MODELS["openrouter"],
                        value="anthropic/claude-opus-4-6:online",
                        label="Model",
                        allow_custom_value=True,
                        info="Select or type a model name. ':online' = web-enabled",
                    )
                    api_key_input = gr.Textbox(
                        label="API Key",
                        placeholder="sk-... (OpenRouter) or AIza... (Gemini)",
                        type="password",
                        visible=True,
                        info="Required for OpenRouter and Gemini",
                    )

                with gr.Group():
                    gr.Markdown("#### Set Configuration")
                    num_tracks_slider = gr.Slider(
                        minimum=2, maximum=20, value=5, step=1,
                        label="Number of Tracks",
                    )
                    with gr.Row():
                        bpm_min_slider = gr.Slider(
                            minimum=60, maximum=200, value=80, step=5,
                            label="BPM Min",
                        )
                        bpm_max_slider = gr.Slider(
                            minimum=60, maximum=200, value=140, step=5,
                            label="BPM Max",
                        )
                    duration_slider = gr.Slider(
                        minimum=15, maximum=180, value=60, step=5,
                        label="Duration per Track (seconds)",
                    )
                    genre_select = gr.Dropdown(
                        choices=AVAILABLE_GENRES,
                        value=[],
                        label="Genre Constraints (optional)",
                        multiselect=True,
                        info="Leave empty for any genre",
                    )
                    mood_dropdown = gr.Dropdown(
                        choices=[e.value for e in MoodProgression],
                        value=MoodProgression.BUILDING.value,
                        label="Mood Progression",
                        info="How the energy evolves across the set",
                    )
                    instrumental_check = gr.Checkbox(
                        label="Instrumental Only",
                        value=False,
                    )

                with gr.Row():
                    plan_btn = gr.Button("📋 Plan Setlist", variant="secondary")
                    generate_btn = gr.Button("🎵 Generate Set", variant="primary")
                    cancel_btn = gr.Button("⏹ Cancel", variant="stop")

            # ---- Right column: Output ----
            with gr.Column(scale=1):
                status_display = gr.Markdown(
                    value="*Ready. Enter a vibe description and click Plan Setlist or Generate Set.*",
                    label="Status",
                )
                setlist_display = gr.Markdown(
                    value="",
                    label="Setlist Plan",
                )
                # Audio outputs — show up to 20 tracks
                audio_outputs = []
                with gr.Group():
                    gr.Markdown("#### Generated Tracks")
                    for i in range(20):
                        audio = gr.Audio(
                            label=f"Track {i + 1}",
                            visible=(i < 5),  # Show first 5 by default
                            interactive=False,
                        )
                        audio_outputs.append(audio)

                # Interactive redirect
                with gr.Group():
                    gr.Markdown("#### 🎛️ Live Redirect")
                    redirect_input = gr.Textbox(
                        label="Redirect the DJ",
                        placeholder="e.g., 'make it more jazzy' or 'add vocals' or 'bring the energy down'",
                        lines=1,
                    )
                    redirect_btn = gr.Button("🔄 Send Redirect", variant="secondary")

        # ---- State ----
        session_state = gr.State(value=None)
        plan_state = gr.State(value=None)

        # ---- Event wiring ----

        # Provider change → update model list + api key visibility
        provider_dropdown.change(
            fn=_update_model_choices,
            inputs=[provider_dropdown],
            outputs=[model_dropdown],
        )
        provider_dropdown.change(
            fn=_needs_api_key,
            inputs=[provider_dropdown],
            outputs=[api_key_input],
        )

        def _plan_only(
            vibe, provider, model, api_key,
            num_tracks, bpm_min_val, bpm_max_val, duration, genres,
            mood, instrumental,
        ):
            """Plan setlist without generating."""
            if not vibe.strip():
                return "*⚠️ Please enter a vibe description.*", "", None
            bpm_min_val = int(bpm_min_val) if bpm_min_val else 80
            bpm_max_val = int(bpm_max_val) if bpm_max_val else 140

            config = DJConfig(
                provider=LLMProvider(provider),
                model=model or DEFAULT_MODELS.get(LLMProvider(provider), ""),
                api_key=api_key,
                num_tracks=int(num_tracks),
                bpm_min=bpm_min_val,
                bpm_max=bpm_max_val,
                genre_constraints=genres or [],
                duration_per_track=int(duration),
                mood_progression=MoodProgression(mood),
                instrumental=instrumental,
            )
            try:
                plan = plan_setlist(config, vibe)
                plan_data = {
                    "narrative": plan.overall_narrative,
                    "tracks": [
                        {
                            "index": t.index,
                            "title": t.title,
                            "genre": t.genre,
                            "bpm": t.bpm,
                            "key": t.key,
                            "energy": t.energy,
                            "caption": t.caption,
                        }
                        for t in plan.tracks
                    ],
                }
                display = _format_setlist_display(plan_data)
                return f"*✅ Setlist planned: {len(plan.tracks)} tracks*", display, plan_data
            except Exception as e:
                return f"*❌ Planning failed: {e}*", "", None

        plan_btn.click(
            fn=_plan_only,
            inputs=[
                vibe_input, provider_dropdown, model_dropdown, api_key_input,
                num_tracks_slider, bpm_min_slider, bpm_max_slider, duration_slider, genre_select,
                mood_dropdown, instrumental_check,
            ],
            outputs=[status_display, setlist_display, plan_state],
        )

        def _generate_set(
            vibe, provider, model, api_key,
            num_tracks, bpm_min_val, bpm_max_val, duration, genres,
            mood, instrumental, existing_plan,
        ):
            """Plan (if needed) and generate the full set.

            Uses Gradio generator pattern to yield updates as tracks complete.
            """
            if not vibe.strip():
                # Return initial state + all audio slots empty
                outputs = ["*⚠️ Please enter a vibe description.*", ""]
                outputs += [gr.update() for _ in range(20)]
                yield outputs
                return

            bpm_min_val = int(bpm_min_val) if bpm_min_val else 80
            bpm_max_val = int(bpm_max_val) if bpm_max_val else 140

            config = DJConfig(
                provider=LLMProvider(provider),
                model=model or DEFAULT_MODELS.get(LLMProvider(provider), ""),
                api_key=api_key,
                num_tracks=int(num_tracks),
                bpm_min=bpm_min_val,
                bpm_max=bpm_max_val,
                genre_constraints=genres or [],
                duration_per_track=int(duration),
                mood_progression=MoodProgression(mood),
                instrumental=instrumental,
            )

            engine = DJEngine(dit_handler, llm_handler, config)

            # Plan
            status_text = "*🎧 Planning setlist...*"
            outputs = [status_text, ""]
            outputs += [gr.update() for _ in range(20)]
            yield outputs

            try:
                session = engine.create_session(vibe)
            except Exception as e:
                outputs = [f"*❌ Planning failed: {e}*", ""]
                outputs += [gr.update() for _ in range(20)]
                yield outputs
                return

            plan_data = {
                "narrative": session.setlist.overall_narrative,
                "tracks": [
                    {
                        "index": t.index,
                        "title": t.title,
                        "genre": t.genre,
                        "bpm": t.bpm,
                        "key": t.key,
                        "energy": t.energy,
                        "caption": t.caption,
                    }
                    for t in session.setlist.tracks
                ],
            }
            setlist_md = _format_setlist_display(plan_data)

            total = len(session.setlist.tracks)

            # Show the correct number of audio slots
            audio_updates = []
            for j in range(20):
                if j < total:
                    track = session.setlist.tracks[j]
                    audio_updates.append(gr.update(
                        visible=True,
                        label=f"Track {j + 1}: {track.title} ({track.genre}, {track.bpm} BPM)",
                        value=None,
                    ))
                else:
                    audio_updates.append(gr.update(visible=False, value=None))

            status_text = f"*🎵 Generating track 1/{total}...*"
            yield [status_text, setlist_md] + audio_updates

            # Generate tracks one by one
            for generated in engine.generate_set(session):
                idx = generated.plan.index
                # Update the audio output for this track
                audio_updates_partial = [gr.update() for _ in range(20)]
                if idx < 20 and generated.audio_path:
                    audio_updates_partial[idx] = gr.update(value=generated.audio_path)

                next_idx = idx + 2  # 1-based, next track
                if next_idx <= total:
                    status_text = f"*🎵 Generated track {idx + 1}/{total} ({generated.generation_time:.1f}s). Generating track {next_idx}/{total}...*"
                else:
                    status_text = f"*✅ Set complete! {total} tracks generated.*"

                yield [status_text, setlist_md] + audio_updates_partial

        generate_btn.click(
            fn=_generate_set,
            inputs=[
                vibe_input, provider_dropdown, model_dropdown, api_key_input,
                num_tracks_slider, bpm_min_slider, bpm_max_slider, duration_slider, genre_select,
                mood_dropdown, instrumental_check, plan_state,
            ],
            outputs=[status_display, setlist_display] + audio_outputs,
        )

    # Return components dict for external event handler wiring if needed
    return {
        "vibe_input": vibe_input,
        "provider_dropdown": provider_dropdown,
        "model_dropdown": model_dropdown,
        "api_key_input": api_key_input,
        "num_tracks_slider": num_tracks_slider,
        "bpm_min_slider": bpm_min_slider,
        "bpm_max_slider": bpm_max_slider,
        "duration_slider": duration_slider,
        "genre_select": genre_select,
        "mood_dropdown": mood_dropdown,
        "instrumental_check": instrumental_check,
        "plan_btn": plan_btn,
        "generate_btn": generate_btn,
        "cancel_btn": cancel_btn,
        "redirect_input": redirect_input,
        "redirect_btn": redirect_btn,
        "status_display": status_display,
        "setlist_display": setlist_display,
        "audio_outputs": audio_outputs,
        "session_state": session_state,
        "plan_state": plan_state,
    }
