# WA Transcript App

WA Transcript App is a local-first web app for turning voice notes or uploaded audio into readable text, then refining and translating that text with in-browser AI models.

## What it does

- Upload audio files or record directly in the browser
- Transcribe speech to English text
- Refine rough ASR output into more legible English with optional context notes
- Translate refined text to Japanese
- Save transcript history locally in your browser

## Tech stack and tools used

- `React` + `TypeScript` for the UI and app logic
- `Vite` for local development and builds
- `@xenova/transformers` (Transformers.js) for in-browser AI pipelines
- Browser APIs: `MediaRecorder`, `IndexedDB`, and Cache Storage
- `ESLint` for linting

## Local setup

### Prerequisites

- Node.js 20+ recommended
- npm 10+ recommended

### Install and run

```bash
npm install
npm run dev
```

Then open the local URL printed by Vite (usually `http://localhost:5173`).

### Production build

```bash
npm run build
npm run preview
```

### Lint

```bash
npm run lint
```

## Notes about AI model loading

- Models are downloaded from Hugging Face the first time they are used.
- Cleanup/refinement tries `Xenova/flan-t5-large` first, then falls back to `Xenova/flan-t5-base` if unauthorized or unavailable.
- Translation uses `Xenova/opus-mt-en-jap` sentence-by-sentence.
- First run can be slow while models download; later runs are faster once cached.

## Privacy

- Audio and transcript history are stored locally in your browser.
- No backend server is required for normal use.
