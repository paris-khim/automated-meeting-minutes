# Automated Meeting Minutes (MOM) ðŸŽ™ï¸

**Winner of HackerEarth International Hackathon.** An end-to-end AI pipeline that processes multi-speaker audio to generate structured meeting summaries, action items, and speaker-wise transcripts.

## ðŸ— System Architecture
1. **Audio Pre-processing:** Noise reduction and format normalization.
2. **Speech Diarization:** Identifying "Who spoke when" using `pyannote-audio`.
3. **Speech-to-Text (STT):** High-accuracy transcription using OpenAI Whisper.
4. **MOM Generation:** LLM-based summarization to extract key decisions and tasks.

## ðŸš€ Quick Start
```bash
python main.py --audio meeting_record.mp3 --output summary.json
```

## ðŸ’» Core Logic

### `diarizer.py`
```python
from pyannote.audio import Pipeline

def run_diarization(audio_file, token):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
    diarization = pipeline(audio_file)
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
```

### `summarizer.py`
```python
import openai

def generate_mom(transcript):
    prompt = f"Extract Action Items and Decisions from this meeting transcript: \n{transcript}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

---
*Bridging corporate productivity with Speech AI.*
