"""
AI DJ Mode for ACE-Step 1.5

Inspired by HeartMuLa's LLM-as-DJ concept.

Uses an LLM (local via Ollama, or cloud via OpenRouter/Gemini) to:
1. Take a high-level vibe description ("chill evening transitioning to party")
2. Plan a sequence of songs with varying styles, BPM, keys
3. Generate each song using ACE-Step with appropriate parameters
4. Handle transitions between tracks
5. Allow real-time user direction ("make it more upbeat", "add vocals")

This module is self-contained and can be toggled on/off in the Gradio UI.
"""

import json
import time
import os
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Generator
from loguru import logger


# ============================================================================
# Configuration
# ============================================================================

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"


class TransitionStyle(str, Enum):
    """How tracks transition into each other."""
    CROSSFADE = "crossfade"
    SILENCE = "silence"
    BEAT_MATCH = "beat_match"


class MoodProgression(str, Enum):
    """How the mood/energy evolves across the set."""
    STATIC = "static"           # Same energy throughout
    BUILDING = "building"       # Gradually increasing energy
    WAVE = "wave"               # Energy rises and falls
    PEAK_VALLEY = "peak_valley" # Alternating high and low energy
    WIND_DOWN = "wind_down"     # Gradually decreasing energy


# Default models per provider
DEFAULT_MODELS = {
    LLMProvider.OLLAMA: "llama3.1:8b",
    LLMProvider.OPENROUTER: "anthropic/claude-opus-4-6:online",
    LLMProvider.GEMINI: "gemini-2.0-flash",
}

# Curated model list per provider (shown in UI dropdown)
PROVIDER_MODELS = {
    LLMProvider.OLLAMA: [
        "llama3.1:8b",
        "llama3.1:70b",
        "qwen3:8b",
        "mistral:7b",
        "gemma3:9b",
    ],
    LLMProvider.OPENROUTER: [
        "anthropic/claude-opus-4-6:online",
        "anthropic/claude-sonnet-4-20250514:online",
        "anthropic/claude-haiku-3-5-20241022:online",
        "google/gemini-2.5-pro-preview:online",
        "google/gemini-2.5-flash-preview:online",
        "meta-llama/llama-3.1-405b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
        "deepseek/deepseek-chat-v3-0324",
    ],
    LLMProvider.GEMINI: [
        "gemini-2.0-flash",
        "gemini-2.5-pro-preview-06-05",
        "gemini-2.5-flash-preview-05-20",
    ],
}

# Curated genre list (subset of what ACE-Step handles well)
AVAILABLE_GENRES = [
    "pop", "rock", "electronic", "hip hop", "r&b", "jazz", "classical",
    "country", "folk", "metal", "punk", "indie", "ambient", "lo-fi",
    "house", "techno", "trance", "drum and bass", "dubstep", "reggae",
    "blues", "soul", "funk", "disco", "latin", "bossa nova", "world",
    "cinematic", "soundtrack", "new age", "chillwave", "synthwave",
    "vaporwave", "trap", "edm", "acoustic", "singer-songwriter",
]

# Valid musical keys for setlist planning
MUSICAL_KEYS = [
    "C major", "C minor", "C# major", "C# minor",
    "D major", "D minor", "Eb major", "Eb minor",
    "E major", "E minor", "F major", "F minor",
    "F# major", "F# minor", "G major", "G minor",
    "Ab major", "Ab minor", "A major", "A minor",
    "Bb major", "Bb minor", "B major", "B minor",
]


@dataclass
class DJConfig:
    """Configuration for the AI DJ session.

    Attributes:
        provider: LLM provider to use for planning.
        model: Model name/id for the selected provider.
        api_key: API key (required for OpenRouter/Gemini, ignored for Ollama).
        ollama_url: Base URL for Ollama server.
        num_tracks: Number of tracks to generate in the set.
        transition_style: How to transition between tracks.
        mood_progression: Energy arc across the set.
        bpm_min: Minimum BPM for generated tracks.
        bpm_max: Maximum BPM for generated tracks.
        genre_constraints: If set, restrict generation to these genres.
        duration_per_track: Default duration per track in seconds.
        crossfade_seconds: Duration of crossfade transitions.
        instrumental: Whether to generate instrumental-only tracks.
        turbo: Whether to use the turbo model (faster, fewer steps).
        inference_steps: Number of diffusion steps per track.
        batch_size: Batch size for generation (1 recommended for DJ mode).
    """
    # LLM settings
    provider: LLMProvider = LLMProvider.OLLAMA
    model: str = ""
    api_key: str = ""
    ollama_url: str = "http://localhost:11434"

    # Set configuration
    num_tracks: int = 5
    transition_style: TransitionStyle = TransitionStyle.CROSSFADE
    mood_progression: MoodProgression = MoodProgression.BUILDING
    bpm_min: int = 80
    bpm_max: int = 140
    genre_constraints: List[str] = field(default_factory=list)
    duration_per_track: int = 60
    crossfade_seconds: float = 3.0
    instrumental: bool = False
    turbo: bool = True
    inference_steps: int = 8
    batch_size: int = 1

    def __post_init__(self):
        if not self.model:
            self.model = DEFAULT_MODELS.get(self.provider, "")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["provider"] = self.provider.value
        d["transition_style"] = self.transition_style.value
        d["mood_progression"] = self.mood_progression.value
        return d


@dataclass
class TrackPlan:
    """Plan for a single track in the setlist.

    Generated by the LLM during setlist planning.
    """
    index: int                          # Track position (0-based)
    title: str = ""                     # Working title
    caption: str = ""                   # Style/description prompt for ACE-Step
    genre: str = ""                     # Primary genre
    bpm: int = 120                      # Target BPM
    key: str = "C major"                # Musical key
    duration: int = 60                  # Duration in seconds
    energy: float = 0.5                 # Energy level 0.0-1.0
    instrumental: bool = False          # Instrumental flag
    lyrics: str = ""                    # Lyrics (empty for instrumental)
    vocal_language: str = "en"          # Vocal language code
    transition_note: str = ""           # How this track transitions to next

    def to_generation_params(self) -> Dict[str, Any]:
        """Convert to kwargs compatible with ACE-Step GenerationParams."""
        return {
            "task_type": "text2music",
            "caption": self.caption,
            "lyrics": self.lyrics if self.lyrics else "[Instrumental]",
            "instrumental": self.instrumental or not self.lyrics,
            "bpm": self.bpm,
            "keyscale": self.key,
            "duration": float(self.duration),
            "vocal_language": self.vocal_language,
        }


@dataclass
class SetlistPlan:
    """Complete setlist plan generated by the LLM."""
    vibe_description: str               # Original user input
    tracks: List[TrackPlan] = field(default_factory=list)
    overall_narrative: str = ""          # LLM's description of the set arc
    created_at: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()


@dataclass
class GeneratedTrack:
    """A track that has been generated by ACE-Step."""
    plan: TrackPlan
    audio_path: str = ""
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DJSession:
    """State for an active DJ session."""
    config: DJConfig
    setlist: Optional[SetlistPlan] = None
    generated_tracks: List[GeneratedTrack] = field(default_factory=list)
    current_track_index: int = 0
    is_generating: bool = False
    is_cancelled: bool = False
    redirect_queue: List[str] = field(default_factory=list)


# ============================================================================
# LLM Client — Thin wrapper for each provider
# ============================================================================

class LLMClient:
    """Unified LLM client supporting Ollama, OpenRouter, and Gemini."""

    def __init__(self, config: DJConfig):
        self.config = config

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request and return the assistant's reply."""
        provider = self.config.provider
        if provider == LLMProvider.OLLAMA:
            return self._chat_ollama(system_prompt, user_prompt)
        elif provider == LLMProvider.OPENROUTER:
            return self._chat_openrouter(system_prompt, user_prompt)
        elif provider == LLMProvider.GEMINI:
            return self._chat_gemini(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    # ---- Ollama ----
    def _chat_ollama(self, system_prompt: str, user_prompt: str) -> str:
        import urllib.request
        url = f"{self.config.ollama_url}/api/chat"
        payload = json.dumps({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.7},
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            raise ConnectionError(f"Failed to reach Ollama at {self.config.ollama_url}: {e}")

    # ---- OpenRouter ----
    def _chat_openrouter(self, system_prompt: str, user_prompt: str) -> str:
        import urllib.request
        if not self.config.api_key:
            raise ValueError("OpenRouter requires an API key. Set it in DJ Config or OPENROUTER_API_KEY env var.")
        api_key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY", "")
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = json.dumps({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/ACE-Step/ACE-Step-1.5",
        })
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenRouter request failed: {e}")
            raise ConnectionError(f"OpenRouter API error: {e}")

    # ---- Google Gemini ----
    def _chat_gemini(self, system_prompt: str, user_prompt: str) -> str:
        import urllib.request
        api_key = self.config.api_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("Gemini requires an API key. Set it in DJ Config or GEMINI_API_KEY env var.")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model}:generateContent?key={api_key}"
        payload = json.dumps({
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.7},
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error(f"Gemini request failed: {e}")
            raise ConnectionError(f"Gemini API error: {e}")


# ============================================================================
# Setlist Planner — Uses LLM to plan the set
# ============================================================================

SETLIST_SYSTEM_PROMPT = """\
You are an expert AI music DJ and setlist planner. You create cohesive, \
emotionally-compelling music sets that flow naturally between tracks.

You understand:
- Musical key relationships (circle of fifths, relative keys)
- BPM transitions (gradual changes are smoother)
- Energy dynamics (building, releasing, creating tension)
- Genre blending (which genres mix well together)
- Mood arc (how to tell a story through a set)

When planning a setlist, consider:
1. Start with the vibe description and mood progression type
2. Map out the energy curve across all tracks
3. Choose keys that flow well together (relative keys, parallel keys, or circle-of-fifths neighbors)
4. Transition BPM gradually (no more than ±20 BPM between adjacent tracks unless intentional)
5. Vary textures — alternate between vocal and instrumental, dense and sparse

You MUST respond with valid JSON only — no markdown, no explanation outside the JSON.
"""

SETLIST_USER_TEMPLATE = """\
Plan a DJ set with the following parameters:

Vibe description: {vibe}
Number of tracks: {num_tracks}
Mood progression: {mood_progression}
BPM range: {bpm_min} - {bpm_max}
Duration per track: {duration}s
Genre constraints: {genres}
Instrumental only: {instrumental}

Respond with a JSON object in this exact format:
{{
  "overall_narrative": "A 1-2 sentence description of how the set flows",
  "tracks": [
    {{
      "index": 0,
      "title": "Working Title",
      "caption": "Detailed style description for the AI music generator (be specific about instruments, mood, texture, tempo feel)",
      "genre": "primary genre",
      "bpm": 120,
      "key": "C major",
      "duration": {duration},
      "energy": 0.5,
      "instrumental": {instrumental_json},
      "lyrics": "",
      "vocal_language": "en",
      "transition_note": "How this transitions to the next track"
    }}
  ]
}}

IMPORTANT:
- "caption" should be rich and descriptive (50-150 words), like a music producer's brief
- BPM must be between {bpm_min} and {bpm_max}
- "key" must be a standard key like "C major", "Ab minor", etc.
- "energy" is 0.0 (calm) to 1.0 (maximum intensity)
- If not instrumental, write short lyrics (4-8 lines) appropriate to the vibe
- Generate exactly {num_tracks} tracks
"""


def _parse_setlist_json(raw: str) -> Dict[str, Any]:
    """Extract and parse JSON from LLM response, handling markdown fences."""
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (with optional language tag)
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        # Remove closing fence
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse setlist JSON: {e}\nRaw response:\n{raw[:500]}")
        raise ValueError(f"LLM returned invalid JSON. Try again or use a different model. Error: {e}")


def plan_setlist(config: DJConfig, vibe: str) -> SetlistPlan:
    """Use the LLM to plan a complete setlist.

    Args:
        config: DJ configuration.
        vibe: Natural language description of desired mood/vibe.

    Returns:
        SetlistPlan with all track plans.
    """
    client = LLMClient(config)

    genres_str = ", ".join(config.genre_constraints) if config.genre_constraints else "any genre"
    user_prompt = SETLIST_USER_TEMPLATE.format(
        vibe=vibe,
        num_tracks=config.num_tracks,
        mood_progression=config.mood_progression.value,
        bpm_min=config.bpm_min,
        bpm_max=config.bpm_max,
        duration=config.duration_per_track,
        genres=genres_str,
        instrumental="true" if config.instrumental else "false",
        instrumental_json="true" if config.instrumental else "false",
    )

    logger.info(f"Planning setlist: '{vibe}' ({config.num_tracks} tracks, {config.mood_progression.value})")
    raw_response = client.chat(SETLIST_SYSTEM_PROMPT, user_prompt)
    data = _parse_setlist_json(raw_response)

    tracks = []
    for t in data.get("tracks", []):
        track = TrackPlan(
            index=t.get("index", len(tracks)),
            title=t.get("title", f"Track {len(tracks) + 1}"),
            caption=t.get("caption", ""),
            genre=t.get("genre", ""),
            bpm=int(t.get("bpm", 120)),
            key=t.get("key", "C major"),
            duration=int(t.get("duration", config.duration_per_track)),
            energy=float(t.get("energy", 0.5)),
            instrumental=t.get("instrumental", config.instrumental),
            lyrics=t.get("lyrics", ""),
            vocal_language=t.get("vocal_language", "en"),
            transition_note=t.get("transition_note", ""),
        )
        # Clamp BPM to configured range
        track.bpm = max(config.bpm_min, min(config.bpm_max, track.bpm))
        tracks.append(track)

    plan = SetlistPlan(
        vibe_description=vibe,
        tracks=tracks,
        overall_narrative=data.get("overall_narrative", ""),
    )
    logger.info(f"Setlist planned: {len(tracks)} tracks — {plan.overall_narrative}")
    return plan


# ============================================================================
# Redirect Planner — Adjusts remaining tracks based on user feedback
# ============================================================================

REDIRECT_SYSTEM_PROMPT = """\
You are an AI DJ adjusting a live set based on audience feedback. \
You will receive the current setlist plan and a redirect instruction \
from the user. Modify the REMAINING tracks (not already generated) \
to accommodate the new direction while maintaining musical coherence.

Respond with valid JSON only — the same format as the original setlist \
but only including the remaining (modified) tracks.
"""

REDIRECT_USER_TEMPLATE = """\
Current setlist narrative: {narrative}

Already generated tracks (DO NOT modify):
{generated_summary}

Remaining tracks to modify:
{remaining_json}

User redirect instruction: "{redirect}"

Respond with JSON containing only the modified remaining tracks:
{{
  "overall_narrative": "Updated narrative reflecting the change",
  "tracks": [ ... modified remaining tracks ... ]
}}
"""


def redirect_setlist(
    config: DJConfig,
    session: DJSession,
    redirect_instruction: str,
) -> SetlistPlan:
    """Modify remaining tracks in the setlist based on user feedback.

    Args:
        config: DJ configuration.
        session: Current DJ session state.
        redirect_instruction: Natural language instruction from user.

    Returns:
        Updated SetlistPlan with modified remaining tracks.
    """
    if not session.setlist:
        raise ValueError("No active setlist to redirect")

    client = LLMClient(config)

    # Summarize already-generated tracks
    generated = session.generated_tracks
    gen_summary = "\n".join(
        f"  Track {t.plan.index + 1}: \"{t.plan.title}\" — {t.plan.genre}, {t.plan.bpm} BPM, {t.plan.key}, energy={t.plan.energy}"
        for t in generated
    ) or "  (none yet)"

    # Get remaining track plans
    remaining_start = session.current_track_index
    remaining = session.setlist.tracks[remaining_start:]
    remaining_json = json.dumps([asdict(t) for t in remaining], indent=2)

    user_prompt = REDIRECT_USER_TEMPLATE.format(
        narrative=session.setlist.overall_narrative,
        generated_summary=gen_summary,
        remaining_json=remaining_json,
        redirect=redirect_instruction,
    )

    logger.info(f"Redirecting setlist: '{redirect_instruction}'")
    raw_response = client.chat(REDIRECT_SYSTEM_PROMPT, user_prompt)
    data = _parse_setlist_json(raw_response)

    # Build updated tracks list: keep generated + new remaining
    updated_tracks = list(session.setlist.tracks[:remaining_start])
    for i, t in enumerate(data.get("tracks", [])):
        track = TrackPlan(
            index=remaining_start + i,
            title=t.get("title", f"Track {remaining_start + i + 1}"),
            caption=t.get("caption", ""),
            genre=t.get("genre", ""),
            bpm=int(t.get("bpm", 120)),
            key=t.get("key", "C major"),
            duration=int(t.get("duration", config.duration_per_track)),
            energy=float(t.get("energy", 0.5)),
            instrumental=t.get("instrumental", config.instrumental),
            lyrics=t.get("lyrics", ""),
            vocal_language=t.get("vocal_language", "en"),
            transition_note=t.get("transition_note", ""),
        )
        track.bpm = max(config.bpm_min, min(config.bpm_max, track.bpm))
        updated_tracks.append(track)

    session.setlist.tracks = updated_tracks
    session.setlist.overall_narrative = data.get(
        "overall_narrative", session.setlist.overall_narrative
    )
    logger.info(f"Setlist redirected: now {len(updated_tracks)} tracks total")
    return session.setlist


# ============================================================================
# DJ Engine — Orchestrates generation
# ============================================================================

class DJEngine:
    """Orchestrates the DJ session: planning, generation, and transitions.

    Usage:
        engine = DJEngine(dit_handler, llm_handler, config)
        session = engine.create_session("chill vibes transitioning to energetic party")

        # Generate tracks one by one (generator for progress tracking)
        for track_info in engine.generate_set(session):
            print(f"Generated: {track_info.plan.title} -> {track_info.audio_path}")

        # Or redirect mid-set
        engine.redirect(session, "make it more jazzy")
    """

    def __init__(self, dit_handler, llm_handler, config: DJConfig):
        """
        Args:
            dit_handler: Initialized AceStepHandler instance.
            llm_handler: Initialized LLMHandler instance (can be None if LM not used).
            config: DJ configuration.
        """
        self.dit_handler = dit_handler
        self.llm_handler = llm_handler
        self.config = config

    def create_session(self, vibe: str) -> DJSession:
        """Plan a setlist and create a new DJ session.

        Args:
            vibe: Natural language vibe description.

        Returns:
            DJSession ready for generation.
        """
        setlist = plan_setlist(self.config, vibe)
        session = DJSession(config=self.config, setlist=setlist)
        return session

    def generate_set(
        self,
        session: DJSession,
        save_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Generator[GeneratedTrack, None, None]:
        """Generate all tracks in the setlist.

        Yields GeneratedTrack objects as each track completes.

        Args:
            session: Active DJ session.
            save_dir: Directory to save audio files. Defaults to ./gradio_outputs/dj_sets/.
            progress_callback: Optional callback(current_index, total, status_message).

        Yields:
            GeneratedTrack for each completed track.
        """
        if not session.setlist:
            raise ValueError("Session has no setlist. Call create_session() first.")

        if save_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_dir = os.path.join(project_root, "gradio_outputs", "dj_sets")
        os.makedirs(save_dir, exist_ok=True)

        session.is_generating = True
        total = len(session.setlist.tracks)

        try:
            for i, track_plan in enumerate(session.setlist.tracks):
                if session.is_cancelled:
                    logger.info("DJ session cancelled by user")
                    break

                # Check for redirect instructions
                if session.redirect_queue:
                    redirect_msg = session.redirect_queue.pop(0)
                    logger.info(f"Processing redirect: {redirect_msg}")
                    session.current_track_index = i
                    redirect_setlist(self.config, session, redirect_msg)
                    # Re-read the (possibly changed) track plan
                    if i < len(session.setlist.tracks):
                        track_plan = session.setlist.tracks[i]
                    else:
                        break

                session.current_track_index = i
                status = f"Generating track {i + 1}/{total}: \"{track_plan.title}\" ({track_plan.genre}, {track_plan.bpm} BPM)"
                logger.info(status)
                if progress_callback:
                    progress_callback(i, total, status)

                # Generate the track using ACE-Step
                generated = self._generate_single_track(track_plan, save_dir, i)
                session.generated_tracks.append(generated)
                yield generated

        finally:
            session.is_generating = False

    def redirect(self, session: DJSession, instruction: str):
        """Queue a redirect instruction for the DJ.

        The redirect will be processed before the next track generation.

        Args:
            session: Active DJ session.
            instruction: Natural language redirect (e.g., "make it more upbeat").
        """
        session.redirect_queue.append(instruction)
        logger.info(f"Redirect queued: '{instruction}'")

    def cancel(self, session: DJSession):
        """Cancel the current DJ session."""
        session.is_cancelled = True
        logger.info("DJ session cancellation requested")

    def _generate_single_track(
        self,
        track_plan: TrackPlan,
        save_dir: str,
        index: int,
    ) -> GeneratedTrack:
        """Generate a single track using ACE-Step.

        Args:
            track_plan: Plan for this track.
            save_dir: Output directory.
            index: Track index (for filename).

        Returns:
            GeneratedTrack with audio path and metadata.
        """
        from acestep.inference import generate_music, GenerationParams, GenerationConfig

        start_time = time.time()

        # Build generation params from the track plan
        gen_kwargs = track_plan.to_generation_params()
        params = GenerationParams(**gen_kwargs)
        params.thinking = self.llm_handler is not None and self.llm_handler.llm_initialized
        params.inference_steps = self.config.inference_steps

        gen_config = GenerationConfig(
            batch_size=self.config.batch_size,
            use_random_seed=True,
            audio_format="flac",
        )

        result = generate_music(
            dit_handler=self.dit_handler,
            llm_handler=self.llm_handler,
            params=params,
            config=gen_config,
            save_dir=save_dir,
        )

        elapsed = time.time() - start_time

        # Get the first audio path from the result
        audio_path = ""
        if result.audios:
            first_audio = result.audios[0]
            if isinstance(first_audio, dict):
                audio_path = first_audio.get("path", first_audio.get("audio_path", ""))
            elif isinstance(first_audio, str):
                audio_path = first_audio

        generated = GeneratedTrack(
            plan=track_plan,
            audio_path=audio_path,
            generation_time=elapsed,
            metadata={
                "generation_params": gen_kwargs,
                "result_status": result.status if hasattr(result, "status") else "ok",
            },
        )
        logger.info(f"Track {index + 1} generated in {elapsed:.1f}s: {audio_path}")
        return generated


# ============================================================================
# Convenience function for Gradio integration
# ============================================================================

def run_dj_session(
    dit_handler,
    llm_handler,
    vibe: str,
    provider: str = "ollama",
    model: str = "",
    api_key: str = "",
    num_tracks: int = 5,
    bpm_min: int = 80,
    bpm_max: int = 140,
    genres: Optional[List[str]] = None,
    duration_per_track: int = 60,
    mood_progression: str = "building",
    instrumental: bool = False,
    progress_callback: Optional[Callable] = None,
) -> Generator[Dict[str, Any], None, None]:
    """High-level entry point for DJ mode.

    Yields dictionaries with track info as each track is generated.
    Designed for easy integration with Gradio event handlers.

    Args:
        dit_handler: Initialized AceStepHandler.
        llm_handler: Initialized LLMHandler.
        vibe: Natural language vibe description.
        provider: LLM provider name.
        model: Model name/id.
        api_key: API key for cloud providers.
        num_tracks: Number of tracks.
        bpm_min: Minimum BPM.
        bpm_max: Maximum BPM.
        genres: Genre constraints.
        duration_per_track: Duration per track in seconds.
        mood_progression: Mood progression style.
        instrumental: Whether to generate instrumental only.
        progress_callback: Optional progress callback.

    Yields:
        Dict with keys: index, title, genre, bpm, key, energy, audio_path, generation_time
    """
    config = DJConfig(
        provider=LLMProvider(provider),
        model=model,
        api_key=api_key,
        num_tracks=num_tracks,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        genre_constraints=genres or [],
        duration_per_track=duration_per_track,
        mood_progression=MoodProgression(mood_progression),
        instrumental=instrumental,
    )

    engine = DJEngine(dit_handler, llm_handler, config)
    session = engine.create_session(vibe)

    # Yield the plan first
    yield {
        "type": "plan",
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

    # Yield each generated track
    for generated in engine.generate_set(session, progress_callback=progress_callback):
        yield {
            "type": "track",
            "index": generated.plan.index,
            "title": generated.plan.title,
            "genre": generated.plan.genre,
            "bpm": generated.plan.bpm,
            "key": generated.plan.key,
            "energy": generated.plan.energy,
            "audio_path": generated.audio_path,
            "generation_time": generated.generation_time,
            "transition_note": generated.plan.transition_note,
        }
