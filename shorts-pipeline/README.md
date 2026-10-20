# shorts-pipeline

Long-form channel videos → ranked, captioned, 9:16 Shorts → YouTube upload queue.
Everything runs locally on this Mac (Whisper on Metal, Ollama for scoring, ffmpeg for rendering).
The only network calls are YouTube metadata (ranking) and the final upload.

## Pipeline

```
catalog  →  rank  →  match  →  transcribe  →  analyze  →  segments  →  render  →  queue  →  upload
yt-dlp      velocity   local      mlx-whisper   energy +    heuristic +   face-track   JSON     Data API v3
metadata    + ratios   files      (word-level)  scene cuts  llama3.1 8b   + captions            (OAuth)
```

* **catalog / rank** – pulls the channel's latest N uploads (metadata only) and scores them:
  view velocity (views/day), total views, like ratio, comment ratio. Weights in `config.yaml`.
* **match** – maps each top-ranked video to a file in `source/` (manifest → YouTube ID in filename → fuzzy title).
* **transcribe** – `whisper-large-v3-turbo` via mlx with word timestamps (CPU faster-whisper fallback).
* **analyze** – audio loudness curve + scene-cut timestamps (hardware-decoded when the codec allows).
* **segments** – generates 28–58s sentence-aligned windows and scores hook / engagement / payoff / length,
  then re-ranks the top 12 with a local LLM (Ollama) for hook strength, retention and standalone clarity.
  Out-points snap to the nearest scene cut.
* **render** – YuNet face tracking → smoothed 9:16 crop that pans within a shot and jumps at cuts;
  falls back to a blurred pillarbox when the subject isn't trackable. Burned-in word-highlight captions,
  a 2.5s hook banner, an end CTA, loudness-normalised audio, 1080×1920 @ 30fps H.264.
* **queue / upload** – resumable uploads via YouTube Data API v3, scheduled `publishAt` spacing, private by default.

## Setup (already done on this machine)

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
brew install ffmpeg ollama && brew services start ollama && ollama pull llama3.1:8b
```

## Usage

```sh
# 1. Rank the channel (metadata only) and show the top 25
./shorts catalog --limit 300 --show 25

# 2. Drop master files into ./source/ then match the ranked list to them
./shorts match                       # add --download to fetch anything missing (1080p)

# 3. Transcribe, score, cut and render the matched videos
./shorts process                     # --only <video_id> to do one; --force to recompute

# or do all of the above in one go
./shorts run --top 10

# Review what will go out
./shorts queue
open output/

# 4. Upload (needs secrets/client_secret.json — see below)
./shorts whoami                      # confirms which channel the token belongs to
./shorts upload --dry-run
./shorts upload --limit 5
```

Test the whole pipeline on arbitrary local files without touching the catalog:

```sh
./shorts process --local --source /path/to/folder
```

## Source files

Put exports in `source/`. Matching order:

1. `source/manifest.csv` – `video_id,filename`
2. the 11-character YouTube ID anywhere in the filename
3. fuzzy match between the YouTube title and the filename (threshold `sources.match_threshold`)

## Upload credentials

1. Google Cloud Console → new project → enable **YouTube Data API v3**.
2. Credentials → OAuth client ID → **Desktop app** → download JSON → save as `secrets/client_secret.json`.
3. `./shorts whoami` opens a browser; sign in with the account that has Manager/Editor access to the
   target channel. The token is cached in `secrets/token.json`. Shorts go to that channel.
4. Quota: 10,000 units/day by default; one upload ≈ 1,600 units (~6/day). Request an increase on the
   project if you need more. The queue just drains over multiple days.

## Outputs

`output/<video_id>/<rank>-<slug>-c<n>.mp4` plus a `.jpg` poster and a `.json` sidecar with timings,
scores, title, description and reframe diagnostics. `output/_sample/` holds a reference render.

## Tuning

Everything lives in `config.yaml`: clip length targets, ranking weights, caption font/size/colours,
crop smoothing, codec (`h264_videotoolbox` is ~5× faster than `libx264` if you need throughput),
LLM model, upload privacy and schedule spacing.

Set `llm.enabled: false` to run on the heuristic scorer alone.
