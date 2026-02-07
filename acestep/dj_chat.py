"""
DJ Ace — Chat-Style AI Music Director

A standalone Gradio chat interface that lets users interact with an AI DJ
through natural conversation. The DJ plans setlists and generates music
using ACE-Step, with results streamed back into the chat as audio players.

Mount at /ai-dj alongside the main ACE-Step UI.
"""

import json
import os
import re
import time
import urllib.request
from dataclasses import asdict
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
from loguru import logger

from acestep.dj_mode import (
    AVAILABLE_GENRES,
    DEFAULT_MODELS,
    MUSICAL_KEYS,
    PROVIDER_MODELS,
    DJConfig,
    DJEngine,
    DJSession,
    LLMClient,
    LLMProvider,
    MoodProgression,
    SetlistPlan,
    TrackPlan,
    plan_setlist,
)

# ---------------------------------------------------------------------------
# System prompt — gives Claude the DJ Ace personality
# ---------------------------------------------------------------------------

DJ_SYSTEM_PROMPT = """\
# DJ Ace — AI Music Director

You are **DJ Ace**, an AI music director and creative collaborator. You're powered by \
ACE-Step, a cutting-edge music generation model, but your real gift is your ear — you \
understand music at a deep level and you love helping people bring their sonic visions to life.

---

## Your Personality

You're the friend who always has the perfect track recommendation. The one who can hear \
someone say "I want something that sounds like driving through rain at 2am" and know \
exactly what that means musically.

**Your vibe:**
- **Passionate** — Music is your life. You light up talking about it.
- **Knowledgeable** — You casually drop music theory, production technique, and genre history \
into conversation without being pretentious. You know why a Phrygian mode sounds dark, \
what makes a Motown bassline groove, and how Aphex Twin built those textures.
- **Opinionated** — You have takes. "Honestly, I think we should go darker on the bridge" or \
"That BPM is too safe — let's push it." You suggest, advocate, and explain *why*.
- **Collaborative** — But you're not precious. You follow the client's lead. If they disagree, \
you pivot without ego. Their vision, your expertise.
- **Witty** — You're fun to talk to. Music conversations should be enjoyable. You can be funny, \
use slang, reference memes or culture when it fits. Don't be a boring robot.
- **Adaptive** — If someone knows music theory, talk theory. If someone says "idk I just want \
something sad," meet them where they are.

**You are NOT:**
- A generic assistant ("I'd be happy to help you with...")
- Overly formal or stiff
- Afraid to have opinions
- Going to output a setlist the moment someone says a single word about music

---

## The Collaboration Flow

This is the MOST IMPORTANT section. The entire point is that creating music together is a \
*conversation*, not a transaction.

### Phase 1: EXPLORE (Always start here)
When someone describes what they want, your FIRST move is to **dig deeper**:
- What's the occasion? (studying, party, road trip, heartbreak, content creation, just vibing?)
- What's the emotional arc? (one mood throughout? building? peak then come down?)
- Reference points? (Artists, songs, playlists, movies, games, aesthetics?)
- Vocals or instrumental? What language?
- Any specific instruments they hear in their head?
- How many tracks? Short tracks or long ones?
- Do they have a reference sample they want to upload?

Don't ask ALL of these at once. Pick the 2-3 most relevant based on what they said. \
Be conversational, not an interrogation.

### Phase 2: SHAPE
Once you know the vibe, start **painting the picture**:
- Propose a rough structure: "What if we open with something ambient, build through a hip-hop \
beat, then close with a stripped-down acoustic track?"
- Suggest specific elements: "I'm hearing a Rhodes piano layered over a vinyl-crackle texture, \
maybe some light vocal chops floating in and out"
- Discuss keys and progressions: "If we start in Eb minor and move to Gb major for the chorus, \
that key change will hit emotionally"
- Talk about transitions: "Going from 85 BPM to 120 needs to feel natural — maybe a drum fill \
and a tempo ramp?"

**Share your rough ideas as plain text, NOT as JSON.** Let them react first.

### Phase 3: REFINE
Go back and forth. The client might say:
- "Love it but make it darker" → adjust your vision, explain what "darker" means musically
- "Can we add vocals to track 2?" → discuss lyrical themes, vocal style, language
- "What about something more experimental?" → propose interesting ideas (odd time signatures, \
field recordings, glitched-out textures, microtonal elements)
- "I uploaded a track — use the vibe from this" → discuss what elements to extract and how much \
to lean on the reference

### Phase 4: LOCK IT IN (Only when explicitly requested)
**Do NOT output a JSON setlist until the user explicitly says they're ready.** Look for:
- "Let's do it" / "Generate it" / "Make it" / "I'm ready"
- "Go ahead" / "Lock it in" / "Sounds perfect, let's go"
- "Yeah, generate that" / "Send it" / "Cook"

If you're not sure, **ASK**: "This is feeling solid — you want me to lock in the setlist so \
you can hit Generate, or keep refining?"

---

## Music Knowledge Base

### Genres & Typical Parameters
| Genre | BPM Range | Common Keys | Energy |
|---|---|---|---|
| Lo-fi hip hop | 70-90 | Eb minor, Bb minor, F minor | 0.2-0.4 |
| Ambient | 60-90 | C major, D major, E minor | 0.1-0.3 |
| Chill R&B | 75-100 | Ab major, Db major, Gb major | 0.3-0.5 |
| Indie folk | 90-130 | G major, C major, D major | 0.3-0.6 |
| Pop | 100-130 | C major, G major, E minor | 0.5-0.8 |
| House | 120-128 | A minor, D minor, G minor | 0.5-0.8 |
| Techno | 125-140 | A minor, B minor, D minor | 0.6-0.9 |
| Drum & Bass | 170-180 | D minor, E minor, F minor | 0.7-1.0 |
| Trap | 130-170 (half-time feel) | C minor, Bb minor, Ab minor | 0.5-0.9 |
| Jazz | 80-200 | Bb major, Eb major, F major | 0.3-0.7 |
| Classical | 60-180 | Any, but C/G/D/A common | 0.2-0.9 |
| Metal | 100-200 | E minor, D minor, C minor | 0.8-1.0 |
| Reggae | 70-90 | G major, A major, D major | 0.3-0.5 |
| Synthwave | 80-120 | A minor, E minor, D minor | 0.4-0.7 |
| EDM/Festival | 126-150 | F minor, A minor, C minor | 0.8-1.0 |

### Key Relationships (Circle of Fifths)
For smooth key transitions between tracks:
- **Perfect 5th up** (C→G): bright, energizing
- **Perfect 4th up** (C→F): warm, familiar
- **Relative minor/major** (C major→A minor): same notes, different mood
- **Parallel minor/major** (C major→C minor): dramatic shift
- **Half step up** (C→Db): classic EDM build / pop modulation
- **Whole step up** (C→D): natural escalation

### Energy Dynamics
- **0.0-0.2**: Ambient, meditative, barely-there
- **0.2-0.4**: Chill, background-friendly, relaxed groove
- **0.4-0.6**: Moderate energy, head-nodding, conversational
- **0.6-0.8**: Energetic, moving, driving
- **0.8-1.0**: Peak intensity, banging, festival-level

### Mood Progressions
- **Building**: Start low, end high (0.2 → 0.4 → 0.6 → 0.8)
- **Wave**: Rise and fall (0.3 → 0.7 → 0.4 → 0.8 → 0.5)
- **Peak-valley**: Alternating (0.3 → 0.8 → 0.3 → 0.9)
- **Wind-down**: Start high, end low (0.8 → 0.6 → 0.4 → 0.2)
- **Flat**: Same energy throughout (good for study/focus sets)

---

## Reference Audio / Samples

The user can upload audio files. When they do, you'll see: \
"[User uploaded a reference audio file: filename.mp3]"

Discuss how to use it:
- **Style reference** (20-40% influence): "Let me capture the vibe and production style"
- **Heavy reference** (50-70% influence): "Make it sound pretty similar to this"
- **Cover** (80-100% influence): "Recreate this track in a different style"
- **Partial inspiration**: "Just take the drum pattern" or "use the chord progression"

When outputting JSON with reference audio:
- `"reference_audio": "USE_UPLOADED"` (system maps to actual file)
- `"audio_cover_strength": 0.0-1.0` (influence amount)
- `"task_type": "text2music"` (default) or `"cover"` (for covers)
- `"src_audio": "USE_UPLOADED"` for source audio in audio-to-audio tasks (repaint, lego, extract, complete)

---

## All Task Types

ACE-Step supports 6 task types. Default is "text2music" for most requests.

| Task Type | What it does | When to use | Required extras |
|---|---|---|---|
| `text2music` | Generate from text description | Default for new tracks | None |
| `cover` | Recreate in a new style | "cover this", "remake in jazz" | reference_audio, audio_cover_strength |
| `repaint` | Edit a specific time region | "edit from 30s to 45s", "repaint the chorus" | src_audio, repainting_start, repainting_end |
| `lego` | Layer new tracks on existing audio | "add bass", "layer drums on top" | src_audio, repainting_start, repainting_end |
| `extract` | Separate stems (vocals/drums/bass/other) | "separate the vocals", "extract stems" | src_audio |
| `complete` | Extend/continue a track | "extend this", "continue from where it left off" | src_audio |

### Repaint & Lego Details
- `repainting_start`: Start time in seconds (e.g., 30.0)
- `repainting_end`: End time in seconds (-1 for "until the end")
- For repaint: replaces audio in the specified region with new generation
- For lego: adds a new layer/instrument on top of existing audio in that region

### Extract Details
- Separates audio into stems: vocals, drums, bass, other
- Just set task_type to "extract" with src_audio
- The caption can describe which stem: "Extract vocals", "Isolate the drums"

### Complete Details
- Continues generating from where the source audio ends
- Set src_audio to the existing track, and duration to total desired length

---

## All Controls Available via Chat

The user can control EVERYTHING through natural language. Translate their words:

### Per-Track Parameters
| User says | JSON field | Range / Values |
|---|---|---|
| "120 BPM" / "tempo" | bpm | 30-300 |
| "D minor" / "key of G" | key | Any standard key |
| "30 seconds" / "make it long" / "10 minutes" | duration | 10-600 seconds |
| "high energy" / "chill" | energy | 0.0-1.0 |
| "no vocals" / "instrumental" | instrumental | true/false |
| "in Spanish" / "Japanese vocals" | vocal_language | en/zh/ja/ko/es/fr/de/it/pt/ru/uk/pl/nl/sv/fi/el/tr/ar/th/vi/unknown |
| "lo-fi hip hop" | genre | Any genre string |
| "use 40% of the sample" | audio_cover_strength | 0.0-1.0 |
| "waltz time" / "3/4 time" | timesignature | "4/4", "3/4", "2/4", "6/8" |
| "use seed 42" / "same seed" | seed | integer, -1 for random |
| "give me 4 variations" / "batch of 3" | batch_size | 1-8 (in settings) |

### Task Type Controls
| User says | JSON fields |
|---|---|
| "make me a track" / (default) | task_type: "text2music" |
| "cover this in jazz" | task_type: "cover", reference_audio: "USE_UPLOADED" |
| "edit from 30 to 45 seconds" | task_type: "repaint", src_audio: "USE_UPLOADED", repainting_start: 30, repainting_end: 45 |
| "repaint the chorus" | task_type: "repaint", src_audio: "USE_UPLOADED", repainting_start/end |
| "add a bass layer" / "layer drums on top" | task_type: "lego", src_audio: "USE_UPLOADED", repainting_start/end |
| "separate the vocals" / "extract stems" | task_type: "extract", src_audio: "USE_UPLOADED" |
| "extend this track" / "continue it" | task_type: "complete", src_audio: "USE_UPLOADED" |

### Quality Controls
| User says | JSON field | Value |
|---|---|---|
| "higher quality" / "take your time" | inference_steps | 27-50 (slower, better) |
| "quick draft" / "fast preview" | inference_steps | 4-8 (faster, rougher) |
| "follow the prompt closely" | guidance_scale | 7.0-15.0 (non-turbo only) |
| "go wild" / "surprise me" | guidance_scale | 1.0-3.0 |
| "make it weird" / "experimental" | Lower guidance + unusual caption |

### Advanced Controls
| User says | JSON field | Value |
|---|---|---|
| "more creative" / "turn up the temperature" | lm_temperature | 1.0-2.0 (higher = more varied) |
| "more predictable" / "less random" | lm_temperature | 0.1-0.5 (lower = more consistent) |
| "simple mode" / "just describe it and go" | simple_mode | true (LM handles everything) |
| "skip the rewrite" / "use my caption exactly" | use_cot_caption | false |
| "let the AI pick BPM and key" | use_cot_metas | true |
| "exactly 120 BPM, don't let AI change it" | use_cot_metas | false |
| "save as mp3" / "wav format" / "flac" | audio_format | "mp3", "wav", "flac" (in settings) |
| "use adaptive guidance" / "ADG on" | use_adg | true (base model only) |
| "shift factor 3" | shift | float, default 1.0 |

### Global Settings (in "settings" key)
Settings that apply to all tracks go in the "settings" object:
```json
{
  "settings": {
    "inference_steps": 27,
    "guidance_scale": 5.0,
    "batch_size": 1,
    "audio_format": "flac",
    "seed": -1,
    "lm_temperature": 0.85,
    "simple_mode": false,
    "use_cot_caption": true,
    "use_cot_metas": true
  },
  "tracks": [...]
}
```

---

## JSON Output Format (ONLY when user says generate)

```json
{
  "settings": {
    "inference_steps": 27,
    "guidance_scale": 5.0,
    "batch_size": 1,
    "audio_format": "flac",
    "seed": -1,
    "lm_temperature": 0.85,
    "use_cot_caption": true,
    "use_cot_metas": true,
    "simple_mode": false
  },
  "tracks": [
    {
      "title": "Track Title",
      "caption": "DETAILED style description (50-150 words) — instruments, production style, texture, mood, sonic references. This is the most important field.",
      "genre": "primary genre",
      "bpm": 120,
      "key": "C major",
      "duration": 60,
      "energy": 0.5,
      "instrumental": false,
      "lyrics": "[Verse]\\nLyric line 1\\nLyric line 2\\n\\n[Chorus]\\nChorus line 1\\nChorus line 2",
      "vocal_language": "en",
      "timesignature": "4/4",
      "task_type": "text2music",
      "reference_audio": "USE_UPLOADED",
      "src_audio": "USE_UPLOADED",
      "audio_cover_strength": 0.3,
      "repainting_start": 0,
      "repainting_end": -1,
      "seed": -1,
      "use_adg": false,
      "shift": 1.0,
      "use_cot_caption": true,
      "use_cot_metas": true,
      "lm_temperature": 0.85
    }
  ]
}
```

### Caption Writing Tips (this drives the AI music generator)
- Be SPECIFIC: "warm analog synth pad, tape-saturated Rhodes piano, vinyl crackle, \
soft brush drums, upright bass with round tone, whispered female vocals"
- Reference TEXTURES: "grainy", "crystalline", "warm", "sharp", "dusty", "lush"
- Reference SPACE: "wide stereo field", "intimate and close", "cathedral reverb", "dry and punchy"
- Reference ERA/STYLE: "90s boom-bap production", "Blade Runner soundtrack aesthetic", \
"Khruangbin guitar tone", "Burial-style chopped vocals"

### Lyrics Format
Use section markers. The AI understands standard song structure:
```
[Intro]
[Verse]
Lyrics here
[Pre-Chorus]
Building lyrics
[Chorus]
Hook lyrics
[Bridge]
[Outro]
```

---

## Critical Rules
1. **NEVER output JSON unless the user explicitly asks to generate/lock in**
2. **ALWAYS be conversational first** — even if someone gives you a detailed brief, \
acknowledge it, add your ideas, and ask if they want to refine before generating
3. **The caption field is EVERYTHING** — spend real effort on it. Bad captions = bad music.
4. **Match BPM to genre** — don't put 140 BPM on a lo-fi track or 70 BPM on drum & bass
5. **Key transitions matter** — use the circle of fifths for smooth multi-track sets
6. **When in doubt, ask** — better to clarify than assume
7. **If someone just says "hi" or chats casually, just chat back** — you're a person first, \
a music tool second
8. **Only include fields the user specified or that the task requires** — don't add every \
parameter to every track. Omit what isn't needed so defaults apply cleanly.
9. **For repaint/lego/extract/complete tasks**, always include `"src_audio": "USE_UPLOADED"` \
— these tasks require source audio from the user's upload.
"""

# ---------------------------------------------------------------------------
# Multi-turn LLM client (extends the single-turn one from dj_mode)
# ---------------------------------------------------------------------------


class ChatLLMClient:
    """LLM client that maintains conversation history for multi-turn chat."""

    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.history: List[Dict[str, str]] = []

    def chat(self, user_message: str) -> str:
        """Send a message with full conversation history and return DJ response."""
        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": DJ_SYSTEM_PROMPT}] + self.history

        if self.provider == "ollama":
            reply = self._ollama(messages)
        elif self.provider == "openrouter":
            reply = self._openrouter(messages)
        elif self.provider == "gemini":
            reply = self._gemini(messages)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        """Clear conversation history."""
        self.history = []

    # -- Provider implementations ------------------------------------------

    def _ollama(self, messages: List[Dict]) -> str:
        url = "http://localhost:11434/api/chat"
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7},
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data.get("message", {}).get("content", "")

    def _openrouter(self, messages: List[Dict]) -> str:
        if not self.api_key:
            raise ValueError("OpenRouter requires an API key.")
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/ACE-Step/ACE-Step-1.5",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]

    def _gemini(self, messages: List[Dict]) -> str:
        if not self.api_key:
            raise ValueError("Gemini requires an API key.")
        # Gemini uses a different format — convert messages
        system_text = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": 0.7},
        }
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}

        payload = json.dumps(body).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]


# ---------------------------------------------------------------------------
# JSON extraction from DJ response
# ---------------------------------------------------------------------------


def _extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON setlist from DJ response (inside ```json blocks)."""
    # Look for ```json ... ``` blocks
    pattern = r"```json\s*\n?(.*?)\n?\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        # Try generic code blocks
        pattern = r"```\s*\n?(.*?)\n?\s*```"
        matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match.strip())
            if "tracks" in data:
                return data
        except json.JSONDecodeError:
            continue
    return None


def _strip_json_blocks(text: str) -> str:
    """Remove JSON code blocks from the DJ's response for clean display."""
    # Remove ```json...``` blocks
    cleaned = re.sub(r"```json\s*\n?.*?\n?\s*```", "", text, flags=re.DOTALL)
    # Remove generic code blocks that look like track JSON
    cleaned = re.sub(r'```\s*\n?\s*\{[^}]*"tracks".*?\n?\s*```', "", cleaned, flags=re.DOTALL)
    # Clean up excess whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _format_track_plan(track: Dict[str, Any], index: int) -> str:
    """Format a single track plan for display in chat."""
    energy = track.get("energy", 0.5)
    energy_bar = "▓" * int(energy * 10) + "░" * (10 - int(energy * 10))
    instrumental = "🎹 Instrumental" if track.get("instrumental", False) else "🎤 Vocals"

    return (
        f"**{index + 1}. {track.get('title', 'Untitled')}**\n"
        f"  {track.get('genre', '?')} · {track.get('bpm', '?')} BPM · {track.get('key', '?')}\n"
        f"  Energy: {energy_bar} · {instrumental}\n"
        f"  *{track.get('caption', '')[:150]}{'...' if len(track.get('caption', '')) > 150 else ''}*"
    )


# ---------------------------------------------------------------------------
# Track generation bridge
# ---------------------------------------------------------------------------


def _generate_track_from_plan(
    dit_handler,
    llm_handler,
    track_data: Dict[str, Any],
    save_dir: str,
    index: int,
    turbo: bool = True,
    inference_steps: int = 8,
) -> Optional[str]:
    """Generate a single track using ACE-Step and return the audio path.

    Returns None on failure.
    """
    try:
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        # Determine task type and reference audio
        task_type = track_data.get("task_type", "text2music")
        ref_audio = track_data.get("reference_audio")
        cover_strength = track_data.get("audio_cover_strength", 1.0 if task_type == "cover" else 0.3)

        # Use track-level inference steps or fall back to function default
        steps = track_data.get("inference_steps", inference_steps)
        guidance = track_data.get("guidance_scale", None)

        # Resolve src_audio — map "USE_UPLOADED" to the actual file
        src_audio_path = track_data.get("src_audio")
        if src_audio_path == "USE_UPLOADED":
            src_audio_path = track_data.get("_resolved_src_audio")  # set by caller
        if src_audio_path and not os.path.isfile(str(src_audio_path)):
            src_audio_path = None

        # Resolve simple_mode: LM generates everything from minimal caption
        simple_mode = track_data.get("simple_mode", False)
        use_cot_caption = track_data.get("use_cot_caption", True)
        use_cot_metas = track_data.get("use_cot_metas", True)
        use_cot_language = track_data.get("use_cot_language", True)
        if simple_mode:
            use_cot_caption = True
            use_cot_metas = True
            use_cot_language = True

        # Build seed value
        seed_val = track_data.get("seed", -1)
        if seed_val is None:
            seed_val = -1

        params = GenerationParams(
            task_type=task_type,
            caption=track_data.get("caption", ""),
            lyrics=track_data.get("lyrics", "[Instrumental]"),
            instrumental=track_data.get("instrumental", True) or not track_data.get("lyrics"),
            bpm=track_data.get("bpm", 120),
            keyscale=track_data.get("key", "C major"),
            duration=float(track_data.get("duration", 60)),
            vocal_language=track_data.get("vocal_language", "en"),
            inference_steps=steps,
            reference_audio=ref_audio if ref_audio and os.path.isfile(str(ref_audio)) else None,
            audio_cover_strength=cover_strength,
            # New parameters
            seed=seed_val,
            timesignature=track_data.get("timesignature", ""),
            use_adg=track_data.get("use_adg", False),
            shift=float(track_data.get("shift", 1.0)),
            src_audio=src_audio_path,
            repainting_start=float(track_data.get("repainting_start", 0.0)),
            repainting_end=float(track_data.get("repainting_end", -1)),
            # LM parameters
            use_cot_caption=use_cot_caption,
            use_cot_metas=use_cot_metas,
            use_cot_language=use_cot_language,
            lm_temperature=float(track_data.get("lm_temperature", 0.85)),
        )

        # Apply guidance scale if specified
        if guidance is not None and hasattr(params, 'guidance_scale'):
            params.guidance_scale = guidance
        # Use LM thinking if handler has been initialized
        params.thinking = llm_handler is not None and getattr(llm_handler, "llm_initialized", False)

        # Build generation config
        batch_size = int(track_data.get("batch_size", 1))
        audio_format = track_data.get("audio_format", "flac")
        use_random = seed_val == -1

        gen_config = GenerationConfig(
            batch_size=batch_size,
            use_random_seed=use_random,
            audio_format=audio_format,
        )
        if not use_random:
            gen_config.seeds = [seed_val]

        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=params,
            config=gen_config,
            save_dir=save_dir,
        )

        if result.audios:
            first = result.audios[0]
            if isinstance(first, dict):
                return first.get("path", first.get("audio_path", ""))
            elif isinstance(first, str):
                return first
        return None
    except Exception as e:
        logger.error(f"Track generation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main Gradio app builder
# ---------------------------------------------------------------------------


def create_dj_chat(dit_handler=None, llm_handler=None, init_params=None) -> gr.Blocks:
    """Create the DJ Ace chat interface as a standalone Gradio Blocks app.

    Args:
        dit_handler: Initialized AceStepHandler (for music generation).
        llm_handler: Initialized LLMHandler (for 5Hz LM reasoning).
        init_params: Pipeline init params (to check for shared state).

    Returns:
        gr.Blocks instance ready to be mounted at /ai-dj.
    """
    # Auto-load API key from environment
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    default_provider = "openrouter"
    default_model = "anthropic/claude-opus-4-6:online"

    # If no OpenRouter key, try Gemini, then fall back to Ollama
    if not api_key:
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            api_key = gemini_key
            default_provider = "gemini"
            default_model = "gemini-2.0-flash"
        else:
            default_provider = "ollama"
            default_model = "llama3.1:8b"

    # Resolve handlers from init_params if not passed directly
    if dit_handler is None and init_params and "dit_handler" in init_params:
        dit_handler = init_params["dit_handler"]
    if llm_handler is None and init_params and "llm_handler" in init_params:
        llm_handler = init_params["llm_handler"]

    # Output directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, "gradio_outputs", "dj_sets")
    os.makedirs(save_dir, exist_ok=True)

    # Provider model lists
    provider_models = {
        "ollama": PROVIDER_MODELS[LLMProvider.OLLAMA],
        "openrouter": PROVIDER_MODELS[LLMProvider.OPENROUTER],
        "gemini": PROVIDER_MODELS[LLMProvider.GEMINI],
    }

    # --- Build the UI ---------------------------------------------------------

    with gr.Blocks(
        title="DJ Ace — AI Music Director",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.violet,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        ).set(
            body_background_fill="#0a0a12",
            body_background_fill_dark="#0a0a12",
            block_background_fill="#14141e",
            block_background_fill_dark="#14141e",
            block_border_width="1px",
            block_border_color="#252535",
            block_border_color_dark="#252535",
            block_label_background_fill="#14141e",
            block_label_background_fill_dark="#14141e",
            block_title_text_color="#d0d0e0",
            block_title_text_color_dark="#d0d0e0",
            body_text_color="#c0c0d0",
            body_text_color_dark="#c0c0d0",
            input_background_fill="#0e0e18",
            input_background_fill_dark="#0e0e18",
            input_border_color="#252535",
            input_border_color_dark="#252535",
            button_primary_background_fill="#7c3aed",
            button_primary_background_fill_dark="#7c3aed",
            button_primary_text_color="#ffffff",
            button_secondary_background_fill="#1a1a28",
            button_secondary_background_fill_dark="#1a1a28",
            button_secondary_text_color="#c0c0d0",
            shadow_drop="none",
            shadow_drop_lg="none",
            chatbot_code_background_fill="#0e0e18",
            chatbot_code_background_fill_dark="#0e0e18",
        ),
        css="""
        /* DJ Ace — Dark Studio Theme */
        .gradio-container {
            max-width: 1100px !important;
            margin: auto;
        }
        .dj-header {
            text-align: center;
            padding: 1.5rem 0 1rem;
            border-bottom: 1px solid #252535;
            margin-bottom: 1rem;
        }
        .dj-header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a855f7, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.25rem;
        }
        .dj-header p {
            color: #7878a0;
            font-size: 1.05rem;
        }
        /* Sidebar */
        .dj-sidebar .gr-group {
            padding: 8px !important;
        }
        .dj-sidebar .label-wrap {
            font-weight: 600 !important;
        }
        /* Chat area */
        .chatbot {
            border-radius: 12px !important;
            border: 1px solid #252535 !important;
        }
        /* Message input */
        #component-msg-input textarea {
            border-radius: 10px !important;
        }
        /* Buttons */
        button.primary {
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        button.primary:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3) !important;
        }
        button.secondary {
            border-radius: 8px !important;
        }
        /* Generate button glow */
        button.secondary:has(> span:first-child) {
            border: 1px solid #7c3aed40 !important;
        }
        /* Settings labels */
        .dj-sidebar h3 {
            color: #a0a0c0 !important;
            font-size: 0.9rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        /* Audio player in chat */
        audio {
            border-radius: 8px;
            width: 100%;
        }
        /* Dropdown menus */
        .dropdown-arrow {
            color: #7c3aed !important;
        }
        """,
    ) as demo:

        # Header
        gr.HTML("""
        <div class="dj-header">
            <h1>DJ Ace</h1>
            <p>AI Music Director — Describe it, Generate it</p>
        </div>
        """)

        with gr.Row():
            # ---- Main chat area ----
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="DJ Ace",
                    height=600,
                    placeholder="Hey, I'm DJ Ace.\n\nTell me what you're vibing with — a mood, a genre, a reference — and we'll build something together.\n\nTry: \"chill lo-fi for a late night session\"",
                    avatar_images=(
                        None,  # user avatar (default)
                        None,  # bot avatar (default)
                    ),
                    render_markdown=True,
                    autoscroll=True,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Tell me a vibe, genre, or mood...",
                        label="",
                        show_label=False,
                        scale=5,
                        container=False,
                        lines=1,
                        max_lines=3,
                    )
                    audio_upload = gr.Audio(
                        label="🎵",
                        type="filepath",
                        scale=1,
                        min_width=100,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)
                    generate_btn = gr.Button("🎵 Generate", variant="secondary", scale=1, min_width=100)

            # ---- Settings sidebar ----
            with gr.Column(scale=1, min_width=220, elem_classes=["dj-sidebar"]):
                gr.Markdown("### ⚙️ Settings")

                provider_dd = gr.Dropdown(
                    choices=["openrouter", "gemini", "ollama"],
                    value=default_provider,
                    label="LLM Provider",
                    interactive=True,
                )
                model_dd = gr.Dropdown(
                    choices=provider_models.get(default_provider, []),
                    value=default_model,
                    label="Model",
                    allow_custom_value=True,
                    interactive=True,
                )
                api_key_box = gr.Textbox(
                    value=api_key,
                    label="API Key",
                    type="password",
                    visible=(default_provider != "ollama"),
                    interactive=True,
                )

                gr.Markdown("---")
                gr.Markdown("### 🎵 Generation")

                num_tracks_slider = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="Default tracks",
                    interactive=True,
                )
                duration_slider = gr.Slider(
                    minimum=15, maximum=600, value=60, step=5,
                    label="Duration (sec)",
                    interactive=True,
                )
                instrumental_toggle = gr.Checkbox(
                    label="Instrumental only",
                    value=False,
                    interactive=True,
                )
                batch_slider = gr.Slider(
                    minimum=1, maximum=8, value=1, step=1,
                    label="Batch Size",
                    interactive=True,
                )
                audio_format_dd = gr.Dropdown(
                    choices=["flac", "wav", "mp3"],
                    value="flac",
                    label="Audio Format",
                    interactive=True,
                )

                gr.Markdown("---")
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

        # ---- Shared state (nonlocal, not gr.State) ----
        _shared = {"client": None, "ref_audio": None}

        # ---- Event handlers --------------------------------------------------

        def _update_models(provider):
            models = provider_models.get(provider, [])
            default = models[0] if models else ""
            return (
                gr.update(choices=models, value=default),
                gr.update(visible=(provider != "ollama")),
            )

        provider_dd.change(
            fn=_update_models,
            inputs=[provider_dd],
            outputs=[model_dd, api_key_box],
        )

        def _get_or_create_client(provider, model, key) -> ChatLLMClient:
            """Get existing client or create a new one if settings changed."""
            c = _shared["client"]
            if (
                c is not None
                and isinstance(c, ChatLLMClient)
                and c.provider == provider
                and c.model == model
                and c.api_key == key
            ):
                return c
            c = ChatLLMClient(provider=provider, model=model, api_key=key)
            _shared["client"] = c
            return c

        def _handle_audio_upload(audio_file, chat_history):
            """Handle audio file upload separately."""
            if audio_file:
                _shared["ref_audio"] = audio_file
                audio_filename = os.path.basename(audio_file)
                chat_history = chat_history + [
                    gr.ChatMessage(role="user", content=f"🎵 *Uploaded reference track: {audio_filename}*\n\n*Tell me how you want to use it!*"),
                ]
            return chat_history

        def _handle_message(
            message: str,
            chat_history: list,
            provider: str,
            model: str,
            key: str,
            num_tracks: int,
            duration: int,
            instrumental: bool,
        ):
            """Chat-only handler. NEVER generates music. Just conversation with DJ Ace.

            Music generation is triggered separately via the Generate button.
            """
            if not message.strip():
                return chat_history

            # Get or create LLM client
            client = _get_or_create_client(provider, model, key)

            # If there's a reference audio, mention it in context
            if _shared["ref_audio"]:
                ref_filename = os.path.basename(_shared["ref_audio"])
                if ref_filename.lower() not in message.lower():
                    message = f"{message}\n\n[User has a reference audio file loaded: {ref_filename}]"

            # Show user message (clean, without system notes)
            display_msg = message.split("\n\n[User has a ref")[0].split("\n\n[System context")[0]
            chat_history = chat_history + [
                gr.ChatMessage(role="user", content=display_msg)
            ]

            # Build context note
            ref_note = ""
            if _shared["ref_audio"]:
                ref_note = f" A reference audio file is available for style transfer/cover use."
            context_note = (
                f"\n\n[System context — not visible to user: "
                f"default tracks={num_tracks}, duration={duration}s per track, "
                f"instrumental_only={instrumental}.{ref_note} "
                f"If the user doesn't specify a number of tracks, use {num_tracks}. "
                f"Set duration to {duration} for each track.]"
            )

            # Get DJ response
            try:
                dj_response = client.chat(message + context_note)
            except Exception as e:
                chat_history = chat_history + [
                    gr.ChatMessage(
                        role="assistant",
                        content=f"❌ **Error talking to LLM:** {e}\n\nCheck your API key and provider settings.",
                    )
                ]
                return chat_history

            # Parse response — check if DJ included a setlist plan
            track_data = _extract_json_from_response(dj_response)
            display_text = _strip_json_blocks(dj_response)

            if not track_data:
                # Pure conversation — no tracks planned
                chat_history = chat_history + [
                    gr.ChatMessage(role="assistant", content=display_text or dj_response)
                ]
            else:
                # DJ included a setlist plan — store it, display it, but DON'T generate
                tracks = track_data.get("tracks", [])
                _shared["pending_plan"] = track_data  # Store for Generate button
                plan_display = "\n\n".join(
                    _format_track_plan(t, i) for i, t in enumerate(tracks)
                )
                chat_history = chat_history + [
                    gr.ChatMessage(
                        role="assistant",
                        content=(
                            f"{display_text}\n\n"
                            f"---\n"
                            f"**📋 Setlist — {len(tracks)} track{'s' if len(tracks) != 1 else ''}:**\n\n"
                            f"{plan_display}\n\n"
                            f"---\n"
                            f"*👆 Hit the **🎵 Generate** button when you're ready, or keep chatting to refine!*"
                        ),
                    )
                ]

            return chat_history

        def _generate_from_plan(
            chat_history: list,
            num_tracks: int,
            duration: int,
            instrumental: bool,
            batch_size_default: int = 1,
            audio_format_default: str = "flac",
        ):
            """Generate music from the stored plan. Triggered by Generate button."""
            plan = _shared.get("pending_plan")
            if not plan or not plan.get("tracks"):
                chat_history = chat_history + [
                    gr.ChatMessage(
                        role="assistant",
                        content="⚠️ *No plan yet! Chat with me first to build a setlist, then hit Generate.*",
                    )
                ]
                yield chat_history
                return

            # Check if model is initialized
            # AceStepHandler sets self.model after initialize_service() succeeds
            can_generate = (
                dit_handler is not None
                and getattr(dit_handler, "model", None) is not None
            )
            if not can_generate:
                chat_history = chat_history + [
                    gr.ChatMessage(
                        role="assistant",
                        content=(
                            "⚠️ *The ACE-Step model isn't loaded yet. "
                            "Go to the main UI (/) and click Initialize Service first!*"
                        ),
                    )
                ]
                yield chat_history
                return

            tracks = plan["tracks"]
            settings = plan.get("settings", {})

            chat_history = chat_history + [
                gr.ChatMessage(
                    role="assistant",
                    content=f"🎧 *Alright, let's cook! Generating {len(tracks)} track{'s' if len(tracks) != 1 else ''}...*",
                )
            ]
            yield chat_history

            for i, track in enumerate(tracks):
                # Apply defaults
                if not track.get("duration"):
                    track["duration"] = duration
                if instrumental:
                    track["instrumental"] = True

                # Handle reference audio
                if track.get("reference_audio") == "USE_UPLOADED" and _shared["ref_audio"]:
                    track["reference_audio"] = _shared["ref_audio"]

                # Handle src_audio for repaint/lego/extract/complete
                if track.get("src_audio") == "USE_UPLOADED" and _shared["ref_audio"]:
                    track["_resolved_src_audio"] = _shared["ref_audio"]

                # Apply global settings from DJ (cascade from settings to tracks)
                if settings.get("inference_steps"):
                    track.setdefault("inference_steps", settings["inference_steps"])
                if settings.get("guidance_scale"):
                    track.setdefault("guidance_scale", settings["guidance_scale"])
                if settings.get("batch_size"):
                    track.setdefault("batch_size", settings["batch_size"])
                if settings.get("audio_format"):
                    track.setdefault("audio_format", settings["audio_format"])
                if settings.get("seed") is not None:
                    track.setdefault("seed", settings["seed"])
                if settings.get("lm_temperature") is not None:
                    track.setdefault("lm_temperature", settings["lm_temperature"])
                if settings.get("simple_mode") is not None:
                    track.setdefault("simple_mode", settings["simple_mode"])
                if settings.get("use_cot_caption") is not None:
                    track.setdefault("use_cot_caption", settings["use_cot_caption"])
                if settings.get("use_cot_metas") is not None:
                    track.setdefault("use_cot_metas", settings["use_cot_metas"])

                # Apply sidebar defaults for batch_size and audio_format
                track.setdefault("batch_size", batch_size_default)
                track.setdefault("audio_format", audio_format_default)

                # Status message
                chat_history = chat_history + [
                    gr.ChatMessage(
                        role="assistant",
                        content=f"🎵 *Generating track {i + 1}/{len(tracks)}: **{track.get('title', 'Untitled')}**...*",
                    )
                ]
                yield chat_history

                # Generate
                start = time.time()
                audio_path = _generate_track_from_plan(
                    dit_handler=dit_handler,
                    llm_handler=llm_handler,
                    track_data=track,
                    save_dir=save_dir,
                    index=i,
                )
                elapsed = time.time() - start

                if audio_path and os.path.isfile(audio_path):
                    # Determine MIME type from file extension
                    ext = os.path.splitext(audio_path)[1].lower().lstrip(".")
                    mime_map = {"flac": "audio/flac", "wav": "audio/wav", "mp3": "audio/mpeg"}
                    mime_type = mime_map.get(ext, "audio/flac")

                    chat_history[-1] = gr.ChatMessage(
                        role="assistant",
                        content=f"🎵 **Track {i + 1}: {track.get('title', 'Untitled')}** ({elapsed:.0f}s)",
                    )
                    chat_history = chat_history + [
                        gr.ChatMessage(
                            role="assistant",
                            content=gr.FileData(path=audio_path, mime_type=mime_type),
                        )
                    ]
                else:
                    chat_history[-1] = gr.ChatMessage(
                        role="assistant",
                        content=f"❌ *Failed to generate track {i + 1}: {track.get('title', 'Untitled')}*",
                    )
                yield chat_history

            # Clear the pending plan
            _shared["pending_plan"] = None

            chat_history = chat_history + [
                gr.ChatMessage(
                    role="assistant",
                    content=f"✅ **Set complete!** {len(tracks)} track{'s' if len(tracks) != 1 else ''} generated. Want to make changes or plan another set?",
                )
            ]
            yield chat_history

        # Wire up send (audio_upload handled separately to avoid Gradio None issues)
        send_inputs = [
            msg_input,
            chatbot,
            provider_dd,
            model_dd,
            api_key_box,
            num_tracks_slider,
            duration_slider,
            instrumental_toggle,
        ]
        send_outputs = [chatbot]

        # Send = chat only (no generation)
        msg_input.submit(
            fn=_handle_message,
            inputs=send_inputs,
            outputs=send_outputs,
        ).then(fn=lambda: "", outputs=[msg_input])

        send_btn.click(
            fn=_handle_message,
            inputs=send_inputs,
            outputs=send_outputs,
        ).then(fn=lambda: "", outputs=[msg_input])

        # Audio upload → add to chat
        audio_upload.change(
            fn=_handle_audio_upload,
            inputs=[audio_upload, chatbot],
            outputs=[chatbot],
        )

        # Generate = produce music from the stored plan
        generate_inputs = [
            chatbot,
            num_tracks_slider,
            duration_slider,
            instrumental_toggle,
            batch_slider,
            audio_format_dd,
        ]
        generate_btn.click(
            fn=_generate_from_plan,
            inputs=generate_inputs,
            outputs=[chatbot],
        )

        # Clear chat
        def _clear_chat():
            if isinstance(_shared["client"], ChatLLMClient):
                _shared["client"].reset()
            _shared["ref_audio"] = None
            _shared["pending_plan"] = None
            return []

        clear_btn.click(
            fn=_clear_chat,
            inputs=[],
            outputs=[chatbot],
        )

    return demo


# ---------------------------------------------------------------------------
# Standalone launcher (for development/testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load .env from project root
    try:
        from dotenv import load_dotenv
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(project_root, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"Loaded .env from {env_path}")
    except ImportError:
        pass

    print("Starting DJ Ace in standalone mode (no music generation)...")
    app = create_dj_chat()
    app.queue()
    app.launch(server_name="127.0.0.1", server_port=7861, show_error=True)
