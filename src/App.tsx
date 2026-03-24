import { useEffect, useMemo, useRef, useState } from 'react'
import './wa-transcript.css'

type TranscriptRecord = {
  id: string
  createdAt: number
  audioName: string
  model: string
  language?: string
  rawTranscript: string
  correctedTranscript?: string
  translatedJa?: string
  contextNotes?: string
  durationSec?: number
}

type AudioLibraryItem = {
  id: string
  name: string
  createdAt: number
  mimeType: string
  durationSec?: number
  sizeBytes: number
}

type StoredAudioRecord = AudioLibraryItem & {
  blob: Blob
}

type TranscriberFn = (
  audio: string,
  options?: {
    language?: string | undefined
    chunk_length_s?: number
    stride_length_s?: number
  },
) => Promise<unknown>

type TextGeneratorFn = (
  text: string,
  options?: { max_new_tokens?: number },
) => Promise<unknown>

type TranslatorRunner = (text: string) => Promise<string>

type TranslationTarget = 'ja'
type SavedView = 'raw' | 'corrected' | 'ja'
type WorkflowStage = 'transcribe' | 'refine' | 'translate'

const STORAGE_KEY = 'wa-transcript:records:v2'
const LEGACY_STORAGE_KEY = 'wa-transcript:records:v1'
const AUDIO_DB_NAME = 'wa-transcript-audio'
const AUDIO_STORE_NAME = 'audio-files'
/** Prefer large when HF serves it; `flan-t5-large` often returns 401 without a token — then we load base. */
const CLEANUP_MODEL_CANDIDATES = ['Xenova/flan-t5-large', 'Xenova/flan-t5-base'] as const
const TRANSLATION_MODEL_ID = 'Xenova/opus-mt-en-jap'
const TRANSLATION_FALLBACK_MODEL_ID = 'Xenova/m2m100_418M'
/** Max transcript characters per chunk (T5 encoder limit ~512 tokens; prompt + chunk must fit). */
const REFINE_CHUNK_MAX_CHARS = 380

const MODEL_OPTIONS: Array<{ id: string; label: string }> = [
  { id: 'Xenova/whisper-tiny.en', label: 'Whisper Tiny (English)' },
  { id: 'Xenova/whisper-tiny', label: 'Whisper Tiny (Auto-detect)' },
  { id: 'Xenova/whisper-small', label: 'Whisper Small (Better)' },
]

function safeUUID() {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function formatDateTime(ts: number) {
  try {
    return new Date(ts).toLocaleString()
  } catch {
    return String(ts)
  }
}

function formatDuration(seconds?: number) {
  if (!seconds || !Number.isFinite(seconds)) return ''
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  if (m <= 0) return `${s}s`
  return `${m}m ${s}s`
}

function formatFileSize(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return ''
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function shortModelName(modelId: string) {
  return modelId.split('/').pop() ?? modelId
}

function pickMediaRecorderMimeType() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/ogg',
  ]
  if (typeof MediaRecorder === 'undefined') return undefined
  for (const c of candidates) {
    if (MediaRecorder.isTypeSupported(c)) return c
  }
  return undefined
}

async function decodeAudioDuration(url: string) {
  return await new Promise<number | undefined>((resolve) => {
    const audio = new Audio(url)
    audio.addEventListener(
      'loadedmetadata',
      () => {
        const duration = audio.duration
        resolve(Number.isFinite(duration) ? duration : undefined)
      },
      { once: true },
    )
    audio.addEventListener('error', () => resolve(undefined), { once: true })
  })
}

function readStringField(obj: Record<string, unknown>, key: string) {
  const value = obj[key]
  return typeof value === 'string' ? value : undefined
}

function readNumberField(obj: Record<string, unknown>, key: string) {
  const value = obj[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function migrateRecord(input: unknown): TranscriptRecord | null {
  if (!input || typeof input !== 'object') return null

  const obj = input as Record<string, unknown>
  const id = readStringField(obj, 'id')
  const audioName = readStringField(obj, 'audioName')
  const model = readStringField(obj, 'model')
  const createdAt = readNumberField(obj, 'createdAt')
  const rawTranscript =
    readStringField(obj, 'rawTranscript') ?? readStringField(obj, 'transcript')

  if (!id || !audioName || !model || createdAt === undefined || !rawTranscript) {
    return null
  }

  return {
    id,
    createdAt,
    audioName,
    model,
    language: readStringField(obj, 'language'),
    rawTranscript,
    correctedTranscript: readStringField(obj, 'correctedTranscript'),
    translatedJa: readStringField(obj, 'translatedJa'),
    contextNotes: readStringField(obj, 'contextNotes'),
    durationSec: readNumberField(obj, 'durationSec'),
  }
}

function getTextFromOutput(output: unknown, key: string) {
  if (Array.isArray(output)) {
    const parts = output
      .map((item) => {
        if (!item || typeof item !== 'object') return undefined
        const value = (item as Record<string, unknown>)[key]
        return typeof value === 'string' ? value : undefined
      })
      .filter((value): value is string => Boolean(value))
    return parts.join('\n').trim()
  }

  if (!output || typeof output !== 'object') return ''
  const value = (output as Record<string, unknown>)[key]
  return typeof value === 'string' ? value.trim() : ''
}

function cleanupPrompt(
  rawTranscript: string,
  contextNotes: string,
  canonicalEntities: string[],
  stronger = false,
) {
  const context = contextNotes.trim() || '(none provided)'
  const canonicalBlock =
    canonicalEntities.length > 0
      ? [
          'Canonical names and terms (use these exact spellings when the audio clearly refers to them):',
          ...canonicalEntities.map((e) => `- ${e}`),
          '',
        ].join('\n')
      : ''

  const fewShot = [
    'Example:',
    'Full context notes: The artist is Gomi Kenji.',
    'Rough: I love Konna Gemji\'s art.',
    'Corrected: I love Gomi Kenji\'s art.',
    '',
  ].join('\n')

  const extra = stronger
    ? [
        '- Rewrite rough phrasing into more natural English where possible.',
        '- If a phrase in the transcript sounds like a canonical name or term above, use that exact spelling.',
        '- If the transcript sounds like a context name or term, replace it with the correct spelling from the notes.',
      ]
    : []

  return [
    'Rewrite this rough English speech-to-text transcript into more natural, readable English.',
    fewShot,
    'Rules:',
    '- Keep the transcript in English.',
    '- Preserve the original meaning.',
    '- Use the full context notes and canonical list to correct likely names, artists, terms, and obvious ASR mistakes.',
    '- Make the wording sound more natural while keeping uncertainty when the meaning is unclear.',
    ...extra,
    '- Do not add facts that are not present in the transcript or context notes.',
    '- If you are unsure, keep the intended meaning but make it easier to read.',
    '- Lightly improve capitalization and punctuation.',
    '- Return only the rewritten transcript.',
    '',
    canonicalBlock,
    `Full context notes:\n${context}`,
    '',
    `Transcript:\n${rawTranscript.trim()}`,
  ].join('\n')
}

function normalizeCleanupText(text: string) {
  return text
    .replace(/^corrected(?: english)? transcript:\s*/i, '')
    .replace(/^output:\s*/i, '')
    .trim()
}

function normalizeTranslationText(text: string) {
  return text
    .replace(/^japanese translation:\s*/i, '')
    .replace(/^translation:\s*/i, '')
    .trim()
}

function normalizeFuzzyText(text: string) {
  return text.toLowerCase().replace(/[^a-z]/g, '')
}

function levenshtein(a: string, b: string) {
  const dp = Array.from({ length: a.length + 1 }, () =>
    new Array<number>(b.length + 1).fill(0),
  )

  for (let i = 0; i <= a.length; i += 1) dp[i][0] = i
  for (let j = 0; j <= b.length; j += 1) dp[0][j] = j

  for (let i = 1; i <= a.length; i += 1) {
    for (let j = 1; j <= b.length; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost,
      )
    }
  }

  return dp[a.length][b.length]
}

/** Structured + heuristic extraction of names/terms from user context (longest-first for glossary). */
function extractCanonicalEntities(contextNotes: string): string[] {
  const raw = contextNotes.trim()
  if (!raw) return []

  const seen = new Set<string>()
  const out: string[] = []

  const add = (s: string) => {
    const t = s.trim().replace(/\s+/g, ' ')
    if (t.length < 2) return
    const key = t.toLowerCase()
    if (seen.has(key)) return
    seen.add(key)
    out.push(t)
  }

  const labelRe =
    /^(?:name|names|artist|artists|title|character|characters|term|terms|note|notes|context|people|person|band|album|series|work|works)\s*[:=-]\s*(.+)$/i
  for (const line of raw.split(/\r?\n/)) {
    const m = line.match(labelRe)
    if (m?.[1]) add(m[1].replace(/^["']|["']$/g, '').trim())
  }

  const qre = /"([^"]+)"|'([^']+)'/g
  let qm: RegExpExecArray | null
  while ((qm = qre.exec(raw)) !== null) {
    const inner = qm[1] ?? qm[2]
    if (inner) add(inner)
  }

  const cueRe =
    /\b(?:name is|(?:called|known as))\s+([A-Za-z0-9][^.\n!?]{0,160})/gi
  let cm: RegExpExecArray | null
  while ((cm = cueRe.exec(raw)) !== null) {
    const fragment = cm[1].trim()
    for (const piece of fragment.split(/,|\band\b/i)) {
      const p = piece.replace(/^(the|a|an)\s+/i, '').trim()
      if (p) add(p)
    }
  }

  const titleMatch = raw.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b/g) ?? []
  const ignored = new Set(['The', 'This', 'That', 'Tokyo', 'There', 'They'])
  for (const t of titleMatch) {
    if (!ignored.has(t)) add(t)
  }

  return out.sort((a, b) => b.length - a.length)
}

function maxRatioForPhraseLength(len: number) {
  if (len <= 4) return 0.34
  if (len <= 8) return 0.42
  return 0.48
}

function replaceTokenPreservingEdges(token: string, replacementCore: string) {
  const leading = token.match(/^[^A-Za-z0-9]*/) ?? ['']
  const trailing = token.match(/[^A-Za-z0-9]*$/) ?? ['']
  return `${leading[0]}${replacementCore}${trailing[0]}`
}

function applyContextGlossary(text: string, contextNotes: string) {
  const terms = extractCanonicalEntities(contextNotes)
  if (!terms.length) return text

  let tokens = text.split(/\s+/)

  for (const term of terms) {
    const termWords = term.split(/\s+/).filter(Boolean)
    const termNormalized = normalizeFuzzyText(term)
    if (!termNormalized) continue

    const n = termWords.length
    const ratioLimit = maxRatioForPhraseLength(termNormalized.length)

    phrase: for (let win = n; win <= Math.min(n + 1, tokens.length); win += 1) {
      for (let i = 0; i <= tokens.length - win; i += 1) {
        const window = tokens.slice(i, i + win)
        const leading = window[0].match(/^[^A-Za-z0-9]*/) ?? ['']
        const trailing = window[window.length - 1].match(/[^A-Za-z0-9]*$/) ?? ['']
        const joined = window
          .map((token) => token.replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, ''))
          .join(' ')
          .trim()

        if (!joined) continue

        const windowNormalized = normalizeFuzzyText(joined)
        if (!windowNormalized) continue

        const distance = levenshtein(windowNormalized, termNormalized)
        const ratio = distance / Math.max(windowNormalized.length, termNormalized.length)

        if (ratio <= ratioLimit) {
          const replacement = `${leading[0]}${term}${trailing[0]}`
          tokens = [...tokens.slice(0, i), replacement, ...tokens.slice(i + win)]
          break phrase
        }
      }
    }

    if (termWords.length >= 2) {
      for (const tw of termWords) {
        const tn = normalizeFuzzyText(tw)
        if (tn.length < 3) continue
        const wordLimit = maxRatioForPhraseLength(tn.length)

        for (let i = 0; i < tokens.length; i += 1) {
          const core = tokens[i].replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, '')
          if (!core) continue
          const wn = normalizeFuzzyText(core)
          if (!wn) continue
          if (Math.abs(wn.length - tn.length) > 3) continue

          const d = levenshtein(wn, tn)
          const r = d / Math.max(wn.length, tn.length)
          if (r <= wordLimit) {
            tokens[i] = replaceTokenPreservingEdges(tokens[i], tw)
          }
        }
      }
    }
  }

  return tokens.join(' ')
}

function splitIntoSentences(text: string): string[] {
  const t = text.trim()
  if (!t) return []

  let parts = t
    .split(/(?<=[.!?])\s+|\n+/)
    .map((s) => s.trim())
    .filter(Boolean)

  if (parts.length === 0) parts = [t]

  const out: string[] = []
  for (const p of parts) {
    if (p.length <= REFINE_CHUNK_MAX_CHARS) {
      out.push(p)
    } else {
      for (let i = 0; i < p.length; i += REFINE_CHUNK_MAX_CHARS) {
        const slice = p.slice(i, i + REFINE_CHUNK_MAX_CHARS).trim()
        if (slice) out.push(slice)
      }
    }
  }
  return out.filter(Boolean)
}

function packSentencesIntoChunks(sentences: string[], maxChars: number): string[][] {
  if (!sentences.length) return []
  const chunks: string[][] = []
  let current: string[] = []
  let size = 0

  for (const s of sentences) {
    const addLen = s.length + (current.length ? 1 : 0)
    if (size + addLen > maxChars && current.length) {
      chunks.push(current)
      current = []
      size = 0
    }
    current.push(s)
    size += addLen
  }
  if (current.length) chunks.push(current)
  return chunks
}

function basicReadabilityPass(text: string) {
  let out = text.replace(/\s+/g, ' ').trim()
  out = out.replace(/\b(\w+)(\s+\1\b)+/gi, '$1')
  out = out.replace(/\bi\b/g, 'I')
  out = out.replace(/\bgonna\b/gi, 'going to')
  out = out.replace(/\bwanna\b/gi, 'want to')
  out = out.replace(/\bjust just\b/gi, 'just')
  out = out.replace(/\s+([,.!?])/g, '$1')
  out = out.replace(/(^|[.!?]\s+)([a-z])/g, (_, start, char) => `${start}${char.toUpperCase()}`)
  return out
}

function maxTokensForText(text: string, ceiling: number) {
  const words = text
    .trim()
    .split(/\s+/)
    .filter(Boolean).length
  return Math.max(96, Math.min(ceiling, words * 3))
}

function openAudioDatabase() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('IndexedDB is not available in this browser.'))
      return
    }

    const request = indexedDB.open(AUDIO_DB_NAME, 1)
    request.onupgradeneeded = () => {
      const db = request.result
      if (!db.objectStoreNames.contains(AUDIO_STORE_NAME)) {
        db.createObjectStore(AUDIO_STORE_NAME, { keyPath: 'id' })
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () =>
      reject(request.error ?? new Error('Failed to open audio database.'))
  })
}

async function listStoredAudioRecords() {
  const db = await openAudioDatabase()
  return await new Promise<StoredAudioRecord[]>((resolve, reject) => {
    const transaction = db.transaction(AUDIO_STORE_NAME, 'readonly')
    const store = transaction.objectStore(AUDIO_STORE_NAME)
    const request = store.getAll()

    request.onsuccess = () => resolve((request.result as StoredAudioRecord[]) ?? [])
    request.onerror = () =>
      reject(request.error ?? new Error('Failed to read audio library.'))

    transaction.oncomplete = () => db.close()
    transaction.onerror = () => db.close()
    transaction.onabort = () => db.close()
  })
}

async function getStoredAudioRecord(id: string) {
  const db = await openAudioDatabase()
  return await new Promise<StoredAudioRecord | null>((resolve, reject) => {
    const transaction = db.transaction(AUDIO_STORE_NAME, 'readonly')
    const store = transaction.objectStore(AUDIO_STORE_NAME)
    const request = store.get(id)

    request.onsuccess = () => resolve((request.result as StoredAudioRecord) ?? null)
    request.onerror = () =>
      reject(request.error ?? new Error('Failed to load audio file.'))

    transaction.oncomplete = () => db.close()
    transaction.onerror = () => db.close()
    transaction.onabort = () => db.close()
  })
}

async function saveStoredAudioRecord(record: StoredAudioRecord) {
  const db = await openAudioDatabase()
  return await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(AUDIO_STORE_NAME, 'readwrite')
    const store = transaction.objectStore(AUDIO_STORE_NAME)
    const request = store.put(record)

    request.onsuccess = () => resolve()
    request.onerror = () =>
      reject(request.error ?? new Error('Failed to save audio file.'))

    transaction.oncomplete = () => db.close()
    transaction.onerror = () => db.close()
    transaction.onabort = () => db.close()
  })
}

export default function App() {
  const [mode, setMode] = useState<'record' | 'upload'>('upload')

  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [audioName, setAudioName] = useState<string>('')
  const [audioDurationSec, setAudioDurationSec] = useState<number | undefined>()
  const [currentAudioId, setCurrentAudioId] = useState<string | null>(null)
  const [audioLibrary, setAudioLibrary] = useState<AudioLibraryItem[]>([])

  const [isRecording, setIsRecording] = useState(false)
  const [recordSeconds, setRecordSeconds] = useState(0)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const recordStreamRef = useRef<MediaStream | null>(null)
  const recordMimeRef = useRef<string | undefined>(undefined)
  const recordChunksRef = useRef<Blob[]>([])
  const cancelRecordingRef = useRef(false)
  const recordTimerRef = useRef<number | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)

  const [selectedModel, setSelectedModel] = useState<string>(
    MODEL_OPTIONS[0]?.id ?? 'Xenova/whisper-tiny.en',
  )
  const [languageHint, setLanguageHint] = useState<string>('')
  const [contextNotes, setContextNotes] = useState<string>('')
  const [translationTarget, setTranslationTarget] = useState<TranslationTarget>('ja')

  const [status, setStatus] = useState<string>('')
  const [rawTranscript, setRawTranscript] = useState<string>('')
  const [correctedTranscript, setCorrectedTranscript] = useState<string>('')
  const [translatedJa, setTranslatedJa] = useState<string>('')
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [isRefining, setIsRefining] = useState(false)
  const [isTranslating, setIsTranslating] = useState(false)

  const transcriberCacheRef = useRef<Record<string, Promise<TranscriberFn>>>({})
  const cleanupCacheRef = useRef<Promise<TextGeneratorFn> | null>(null)
  const translatorCacheRef = useRef<Promise<TranslatorRunner> | null>(null)
  const transformersCacheResetRef = useRef(false)

  const [records, setRecords] = useState<TranscriptRecord[]>([])
  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(null)
  const [selectedSavedView, setSelectedSavedView] = useState<SavedView>('raw')
  const [currentDraftRecordId, setCurrentDraftRecordId] = useState<string | null>(null)
  const [expandedStage, setExpandedStage] = useState<WorkflowStage>('transcribe')

  const selectedRecord = useMemo(() => {
    if (!selectedRecordId) return null
    return records.find((r) => r.id === selectedRecordId) ?? null
  }, [records, selectedRecordId])

  const isBusy = isTranscribing || isRefining || isTranslating

  const previousAudioLibrary = useMemo(
    () => audioLibrary.filter((item) => item.id !== currentAudioId),
    [audioLibrary, currentAudioId],
  )

  const selectedSavedText = useMemo(() => {
    if (!selectedRecord) return ''
    if (selectedSavedView === 'ja') return selectedRecord.translatedJa ?? ''
    if (selectedSavedView === 'corrected') return selectedRecord.correctedTranscript ?? ''
    return selectedRecord.rawTranscript
  }, [selectedRecord, selectedSavedView])

  useEffect(() => {
    try {
      const raw =
        localStorage.getItem(STORAGE_KEY) ?? localStorage.getItem(LEGACY_STORAGE_KEY)
      if (!raw) return

      const parsed = JSON.parse(raw) as unknown[]
      if (!Array.isArray(parsed)) return

      const migrated = parsed
        .map((item) => migrateRecord(item))
        .filter((item): item is TranscriptRecord => item !== null)

      setRecords(migrated)
    } catch {
      // ignore
    }
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(records))
    } catch {
      // ignore
    }
  }, [records])

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl)
    }
  }, [audioUrl])

  useEffect(() => {
    void refreshAudioLibrary()
  }, [])

  async function refreshAudioLibrary() {
    try {
      const items = await listStoredAudioRecords()
      const sorted = items
        .map(({ id, name, createdAt, mimeType, durationSec, sizeBytes }) => ({
          id,
          name,
          createdAt,
          mimeType,
          durationSec,
          sizeBytes,
        }))
        .sort((a, b) => b.createdAt - a.createdAt)
      setAudioLibrary(sorted)
    } catch (err) {
      console.warn('Audio library is unavailable:', err)
    }
  }

  function clearTextOutputs() {
    setRawTranscript('')
    setCorrectedTranscript('')
    setTranslatedJa('')
  }

  function resetCurrentDraft() {
    setCurrentDraftRecordId(null)
  }

  function setTimedStatus(message: string, timeoutMs = 1500) {
    setStatus(message)
    window.setTimeout(() => setStatus(''), timeoutMs)
  }

  async function resetTransformersBrowserCache() {
    if (transformersCacheResetRef.current) return
    transformersCacheResetRef.current = true

    if (typeof caches === 'undefined') return
    try {
      await caches.delete('transformers-cache')
    } catch (err) {
      console.warn('Unable to clear transformers cache:', err)
    }
  }

  async function getPipelineLoader() {
    await resetTransformersBrowserCache()
    const { env, pipeline } = await import('@xenova/transformers')
    env.allowLocalModels = false
    env.useBrowserCache = false
    return pipeline
  }

  async function getTranscriber(modelId: string) {
    if (!transcriberCacheRef.current[modelId]) {
      transcriberCacheRef.current[modelId] = (async () => {
        setStatus('Loading speech-to-text model...')
        try {
          const pipeline = await getPipelineLoader()
          const transcriber = await pipeline(
            'automatic-speech-recognition',
            modelId,
          )
          setStatus('')
          return transcriber as unknown as TranscriberFn
        } catch (err: unknown) {
          delete transcriberCacheRef.current[modelId]
          console.error('Model load failed:', err)
          const maybe = err as Record<string, unknown>
          const rawMessage =
            typeof maybe?.message === 'string' ? maybe.message : String(err)
          const message = rawMessage.includes(`Unexpected token '<'`)
            ? 'Model download failed because the app received HTML instead of a model file. This usually means a hosting rewrite or blocked access to Hugging Face.'
            : rawMessage
          setStatus(`${message} (model download may have been blocked)`)
          throw err
        }
      })()
    }
    return transcriberCacheRef.current[modelId]
  }

  async function getCleanupModel() {
    if (!cleanupCacheRef.current) {
      cleanupCacheRef.current = (async () => {
        const pipeline = await getPipelineLoader()
        let lastErr: unknown
        for (const modelId of CLEANUP_MODEL_CANDIDATES) {
          try {
            setStatus(`Loading cleanup model…`)
            const cleaner = await pipeline('text2text-generation', modelId)
            setStatus('')
            if (modelId !== CLEANUP_MODEL_CANDIDATES[0]) {
              console.warn(
                'Refinement model: using',
                modelId,
                '(large model unavailable — often needs Hugging Face auth or is gated).',
              )
            }
            return cleaner as unknown as TextGeneratorFn
          } catch (err) {
            lastErr = err
            console.warn(`Cleanup model load failed (${modelId}):`, err)
          }
        }
        cleanupCacheRef.current = null
        throw lastErr
      })()
    }
    return cleanupCacheRef.current
  }

  async function getTranslatorModel() {
    if (!translatorCacheRef.current) {
      translatorCacheRef.current = (async () => {
        setStatus('Loading translation model...')
        try {
          const pipeline = await getPipelineLoader()
          try {
            const translator = await pipeline('translation', TRANSLATION_MODEL_ID)
            setStatus('')
            return async (text: string) =>
              normalizeTranslationText(
                getTextFromOutput(
                  await (translator as unknown as (input: string) => Promise<unknown>)(text),
                  'translation_text',
                ),
              )
          } catch (primaryErr) {
            console.warn('Primary translation model failed, using fallback:', primaryErr)
          }

          const fallback = await pipeline('translation', TRANSLATION_FALLBACK_MODEL_ID)
          setStatus('')
          return async (text: string) =>
            normalizeTranslationText(
              getTextFromOutput(
                await (
                  fallback as unknown as (
                    input: string,
                    options?: { src_lang?: string; tgt_lang?: string },
                  ) => Promise<unknown>
                )(text, {
                  src_lang: 'en',
                  tgt_lang: 'ja',
                }),
                'translation_text',
              ),
            )
        } catch (err) {
          translatorCacheRef.current = null
          throw err
        }
      })()
    }
    return translatorCacheRef.current
  }

  async function applyCurrentAudio(record: StoredAudioRecord) {
    if (audioUrl) URL.revokeObjectURL(audioUrl)

    const url = URL.createObjectURL(record.blob)
    setAudioUrl(url)
    setAudioName(record.name)
    setAudioDurationSec(record.durationSec)
    setCurrentAudioId(record.id)
    clearTextOutputs()
    resetCurrentDraft()
    setSelectedRecordId(null)
  }

  async function storeAndSelectAudio(blob: Blob, name: string) {
    if (audioUrl) URL.revokeObjectURL(audioUrl)

    const url = URL.createObjectURL(blob)
    const duration = await decodeAudioDuration(url)
    const record: StoredAudioRecord = {
      id: safeUUID(),
      name,
      createdAt: Date.now(),
      mimeType: blob.type || 'audio/webm',
      durationSec: duration,
      sizeBytes: blob.size,
      blob,
    }

    try {
      await saveStoredAudioRecord(record)
      await refreshAudioLibrary()
    } catch (err) {
      console.warn('Unable to persist audio file:', err)
    }

    setAudioUrl(url)
    setAudioName(record.name)
    setAudioDurationSec(record.durationSec)
    setCurrentAudioId(record.id)
    clearTextOutputs()
    resetCurrentDraft()
    setSelectedRecordId(null)
  }

  async function loadAudioFromLibrary(id: string) {
    try {
      const record = await getStoredAudioRecord(id)
      if (!record) {
        setStatus('That audio file is no longer available.')
        window.setTimeout(() => setStatus(''), 2500)
        await refreshAudioLibrary()
        return
      }
      await applyCurrentAudio(record)
      setTimedStatus(`Loaded audio: ${record.name}`)
    } catch (err) {
      console.error('Failed to load audio from library:', err)
      setStatus('Unable to load that audio file.')
      window.setTimeout(() => setStatus(''), 2500)
    }
  }

  async function transcribeCurrentAudio() {
    if (!audioUrl || !selectedModel) return

    setIsTranscribing(true)
    clearTextOutputs()
    resetCurrentDraft()
    setStatus('Transcribing...')
    let success = false

    try {
      const transcriber = await getTranscriber(selectedModel)
      const out = await transcriber(audioUrl, {
        language: languageHint.trim() ? languageHint.trim() : undefined,
        chunk_length_s: 30,
        stride_length_s: 5,
      })

      const text =
        getTextFromOutput(out, 'text') ||
        getTextFromOutput(out, 'generated_text')

      const finalText = text.trim()
      setRawTranscript(finalText)
      upsertCurrentRecord({ rawTranscript: finalText })
      setStatus('Raw transcript complete. Saved.')
      success = true
    } catch (err: unknown) {
      const maybe = err as Record<string, unknown>
      const message =
        typeof maybe?.message === 'string'
          ? maybe.message
          : String(err) ||
            'Transcription failed. Try a different audio format (mp3/wav/m4a/webm/ogg).'
      console.error('Transcription failed:', err)
      setStatus(`${message} (check console for details)`)
    } finally {
      setIsTranscribing(false)
      window.setTimeout(() => setStatus(''), success ? 2500 : 15000)
    }
  }

  async function refineTranscript() {
    const source = rawTranscript.trim()
    if (!source) return

    setIsRefining(true)
    setCorrectedTranscript('')
    setTranslatedJa('')
    setStatus('Refining transcript with context...')
    let success = false

    try {
      const cleaner = await getCleanupModel()
      const entities = extractCanonicalEntities(contextNotes)
      const glossaryAdjusted = applyContextGlossary(source, contextNotes)
      const heuristicBase = basicReadabilityPass(glossaryAdjusted)

      const sentences = splitIntoSentences(heuristicBase)
      const chunks = packSentencesIntoChunks(sentences, REFINE_CHUNK_MAX_CHARS)
      const chunkGroups = chunks.length ? chunks : [[heuristicBase]]

      async function runRefinePass(stronger: boolean) {
        const parts: string[] = []
        for (const chunk of chunkGroups) {
          const chunkText = chunk.join(' ')
          const prompt = cleanupPrompt(chunkText, contextNotes, entities, stronger)
          const out = normalizeCleanupText(
            getTextFromOutput(
              await cleaner(prompt, {
                max_new_tokens: maxTokensForText(chunkText, 512),
              }),
              'generated_text',
            ),
          )
          parts.push(out.trim() ? out : chunkText)
        }
        return basicReadabilityPass(parts.join(' '))
      }

      let cleaned = await runRefinePass(false)

      if ((!cleaned || cleaned === heuristicBase) && contextNotes.trim()) {
        cleaned = await runRefinePass(true)
      }

      const finalCleaned = basicReadabilityPass(
        applyContextGlossary(cleaned || heuristicBase, contextNotes),
      )

      if (!finalCleaned) {
        throw new Error('Cleanup model returned an empty transcript.')
      }

      setCorrectedTranscript(finalCleaned)
      setStatus(
        finalCleaned === source
          ? 'Refinement complete. No confident context-based edits were made.'
          : 'Context refinement complete.',
      )
      success = true
    } catch (err: unknown) {
      const maybe = err as Record<string, unknown>
      const message =
        typeof maybe?.message === 'string'
          ? maybe.message
          : String(err) || 'Context cleanup failed.'
      console.error('Cleanup failed:', err)
      setStatus(`${message} (raw transcript is still available)`)
    } finally {
      setIsRefining(false)
      window.setTimeout(() => setStatus(''), success ? 3000 : 15000)
    }
  }

  async function translateTranscript() {
    const source = correctedTranscript.trim()
    if (!source) return

    setIsTranslating(true)
    setTranslatedJa('')
    setStatus('Translating to Japanese...')
    let success = false

    try {
      const translator = await getTranslatorModel()
      const sentences = splitIntoSentences(source)
      const toTranslate = sentences.length ? sentences : [source]
      const parts: string[] = []

      for (const sentence of toTranslate) {
        const s = sentence.trim()
        if (!s) continue
        const piece = await translator(s)
        if (piece.trim()) parts.push(piece.trim())
      }

      const translated = parts.join('\n')

      if (!translated) {
        throw new Error('Translation model returned an empty result.')
      }

      setTranslatedJa(translated)
      setStatus('Japanese translation complete.')
      success = true
    } catch (err: unknown) {
      const maybe = err as Record<string, unknown>
      const message =
        typeof maybe?.message === 'string'
          ? maybe.message
          : String(err) || 'Translation failed.'
      console.error('Translation failed:', err)
      setStatus(`${message} (corrected English is still available)`)
    } finally {
      setIsTranslating(false)
      window.setTimeout(() => setStatus(''), success ? 3000 : 15000)
    }
  }

  async function startRecording() {
    try {
      clearTextOutputs()
      resetCurrentDraft()
      setSelectedRecordId(null)
      setStatus('')
      cancelRecordingRef.current = false

      if (!navigator.mediaDevices?.getUserMedia) {
        setStatus('Microphone not supported in this browser.')
        return
      }

      const mimeType = pickMediaRecorderMimeType()
      recordMimeRef.current = mimeType
      recordChunksRef.current = []

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      recordStreamRef.current = stream

      const recorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType } : undefined,
      )
      mediaRecorderRef.current = recorder

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) recordChunksRef.current.push(event.data)
      }

      recorder.onstop = async () => {
        if (cancelRecordingRef.current) {
          recordChunksRef.current = []
          recordStreamRef.current?.getTracks().forEach((t) => t.stop())
          recordStreamRef.current = null
          setStatus('')
          return
        }

        const chunks = recordChunksRef.current
        const mime = recordMimeRef.current || 'audio/webm'
        const blob = new Blob(chunks, { type: mime })
        await storeAndSelectAudio(blob, 'recording.webm')

        recordStreamRef.current?.getTracks().forEach((t) => t.stop())
        recordStreamRef.current = null
      }

      recorder.start()
      setIsRecording(true)
      setRecordSeconds(0)

      if (recordTimerRef.current) window.clearInterval(recordTimerRef.current)
      recordTimerRef.current = window.setInterval(() => {
        setRecordSeconds((s) => s + 1)
      }, 1000)
    } catch (err: unknown) {
      const maybe = err as Record<string, unknown>
      const message =
        typeof maybe?.message === 'string'
          ? maybe.message
          : String(err) || 'Failed to start recording.'
      setStatus(message)
    }
  }

  function stopRecording() {
    try {
      mediaRecorderRef.current?.stop()
      setIsRecording(false)
      if (recordTimerRef.current) window.clearInterval(recordTimerRef.current)
      recordTimerRef.current = null
      setStatus('Processing recording...')
    } catch (err: unknown) {
      const maybe = err as Record<string, unknown>
      const message =
        typeof maybe?.message === 'string'
          ? maybe.message
          : String(err) || 'Failed to stop recording.'
      setStatus(message)
    }
  }

  function clearAudio() {
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setAudioUrl(null)
    setAudioName('')
    setAudioDurationSec(undefined)
    setCurrentAudioId(null)
    clearTextOutputs()
    resetCurrentDraft()
    setSelectedRecordId(null)
    setStatus('')
  }

  async function onUploadFile(file: File) {
    setMode('upload')
    clearTextOutputs()
    resetCurrentDraft()
    setSelectedRecordId(null)
    setStatus('')
    await storeAndSelectAudio(file, file.name)
  }

  function upsertCurrentRecord(partial: {
    rawTranscript?: string | undefined
    correctedTranscript?: string | undefined
    translatedJa?: string | undefined
    contextNotes?: string | undefined
  }) {
    const raw = (partial.rawTranscript ?? rawTranscript).trim()
    if (!raw || !audioUrl) return

    const nextId = currentDraftRecordId ?? safeUUID()
    const nextRecord: TranscriptRecord = {
      id: nextId,
      createdAt: Date.now(),
      audioName: audioName || 'audio',
      model: selectedModel,
      language: languageHint.trim() ? languageHint.trim() : undefined,
      rawTranscript: raw,
      correctedTranscript: partial.correctedTranscript,
      translatedJa: partial.translatedJa,
      contextNotes: partial.contextNotes,
      durationSec: audioDurationSec,
    }

    setRecords((prev) => {
      const existing = prev.find((item) => item.id === nextId)
      if (!existing) return [nextRecord, ...prev].slice(0, 50)

      const updated: TranscriptRecord = {
        ...existing,
        ...nextRecord,
        correctedTranscript:
          partial.correctedTranscript ?? existing.correctedTranscript,
        translatedJa: partial.translatedJa ?? existing.translatedJa,
        contextNotes: partial.contextNotes ?? existing.contextNotes,
      }

      return prev.map((item) => (item.id === nextId ? updated : item))
    })

    setCurrentDraftRecordId(nextId)
    setSelectedRecordId(nextId)
    setSelectedSavedView(
      partial.translatedJa ? 'ja' : partial.correctedTranscript ? 'corrected' : 'raw',
    )
  }

  function saveRefinement() {
    const corrected = correctedTranscript.trim()
    if (!corrected) return
    upsertCurrentRecord({
      correctedTranscript: corrected,
      contextNotes: contextNotes.trim() || undefined,
    })
    setTimedStatus('Saved refinement on this record.')
  }

  function saveTranslation() {
    const translated = translatedJa.trim()
    if (!translated) return
    upsertCurrentRecord({
      correctedTranscript: correctedTranscript.trim() || undefined,
      translatedJa: translated,
      contextNotes: contextNotes.trim() || undefined,
    })
    setTimedStatus('Saved translation on this record.')
  }

  function loadRecord(rec: TranscriptRecord, view: SavedView = 'raw') {
    setSelectedRecordId(rec.id)
    setSelectedSavedView(view)
    setCurrentDraftRecordId(rec.id)
    setRawTranscript(rec.rawTranscript)
    setCorrectedTranscript(rec.correctedTranscript ?? '')
    setTranslatedJa(rec.translatedJa ?? '')
    setContextNotes(rec.contextNotes ?? '')
    setTimedStatus(`Loaded saved record: ${rec.audioName}`)
  }

  function deleteRecord(id: string) {
    setRecords((prev) => prev.filter((r) => r.id !== id))
    if (selectedRecordId === id) setSelectedRecordId(null)
    if (currentDraftRecordId === id) setCurrentDraftRecordId(null)
  }

  async function copyText(text: string, label: string) {
    if (!text.trim()) return

    try {
      await navigator.clipboard.writeText(text.trim())
      setTimedStatus(`Copied ${label}.`)
    } catch {
      setStatus('Copy failed. Select the text manually.')
      window.setTimeout(() => setStatus(''), 2500)
    }
  }

  return (
    <div id="wa-app">
      <header className="wa-header">
        <div>
          <div className="wa-title">WhatsApp Voice Transcriber</div>
          <div className="wa-subtitle">Transcribe, refine, translate.</div>
        </div>
        <div className="wa-pill">
          STT <span className="wa-pillStrong">{shortModelName(selectedModel)}</span> · Refine{' '}
          <span className="wa-pillStrong">flan-t5</span> · Translate{' '}
          <span className="wa-pillStrong">opus-mt / m2m100</span>
        </div>
      </header>

      <main className="wa-grid">
        <section className="wa-card">
          <div className="wa-cardTitleRow">
            <div>
              <div className="wa-sectionKicker">01 Audio</div>
              <div className="wa-cardTitle">Source</div>
            </div>
            <div className="wa-seg">
              <button
                type="button"
                className={mode === 'upload' ? 'wa-segBtn wa-segBtnActive' : 'wa-segBtn'}
                onClick={() => setMode('upload')}
              >
                Upload
              </button>
              <button
                type="button"
                className={mode === 'record' ? 'wa-segBtn wa-segBtnActive' : 'wa-segBtn'}
                onClick={() => setMode('record')}
              >
                Record
              </button>
            </div>
          </div>

          {mode === 'upload' ? (
            <div className="wa-stack">
              <label
                className={isDragOver ? 'wa-drop wa-dropDrag' : 'wa-drop'}
                onDragOver={(e) => {
                  e.preventDefault()
                  setIsDragOver(true)
                }}
                onDragEnter={(e) => {
                  e.preventDefault()
                  setIsDragOver(true)
                }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault()
                  setIsDragOver(false)
                  const file = e.dataTransfer.files?.[0]
                  if (file) void onUploadFile(file)
                }}
              >
                <div className="wa-dropTitle">Drop audio</div>
                <div className="wa-dropHint">or click to choose</div>
                <input
                  type="file"
                  accept="audio/*,.m4a,.mp3,.wav,.webm,.ogg,.opus"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) void onUploadFile(file)
                  }}
                />
              </label>
              <div className="wa-muted">Saved locally in this browser.</div>
            </div>
          ) : (
            <div className="wa-stack">
              <div className="wa-row">
                {!isRecording ? (
                  <button
                    type="button"
                    className="wa-primary"
                    onClick={() => void startRecording()}
                    disabled={isBusy}
                  >
                    Start recording
                  </button>
                ) : (
                  <button type="button" className="wa-danger" onClick={stopRecording}>
                    Stop ({recordSeconds}s)
                  </button>
                )}
                <button
                  type="button"
                  className="wa-ghost"
                  onClick={() => {
                    cancelRecordingRef.current = true
                    mediaRecorderRef.current?.stop()
                    setIsRecording(false)
                    if (recordTimerRef.current) window.clearInterval(recordTimerRef.current)
                    recordTimerRef.current = null
                    setStatus('Cancelled.')
                    window.setTimeout(() => setStatus(''), 1000)
                  }}
                  disabled={!isRecording}
                >
                  Cancel
                </button>
              </div>
              <div className="wa-muted">Saved locally in this browser.</div>
            </div>
          )}

          <div className="wa-audioPanel">
            <div className="wa-subCard">
              <div className="wa-subCardTitle">Current</div>
              {!audioUrl ? (
                <div className="wa-empty">No audio selected.</div>
              ) : (
                <>
                  <div className="wa-audioMeta">
                    <div className="wa-audioName" title={audioName}>
                      {audioName}
                    </div>
                    <div className="wa-audioRight">
                      <span className="wa-tag">{formatDuration(audioDurationSec)}</span>
                      <button type="button" className="wa-ghostSmall" onClick={clearAudio}>
                        Clear
                      </button>
                    </div>
                  </div>
                  <audio className="wa-audioEl" controls src={audioUrl} />
                </>
              )}
            </div>

            <div className="wa-subCard">
              <div className="wa-subCardTitle">Library</div>
              {previousAudioLibrary.length === 0 ? (
                <div className="wa-emptySmall">Previous audio appears here.</div>
              ) : (
                <div className="wa-historyList">
                  {previousAudioLibrary.map((item) => (
                    <div key={item.id} className="wa-historyItem">
                      <div className="wa-historyTop">
                        <div className="wa-historyName" title={item.name}>
                          {item.name}
                        </div>
                        <button
                          type="button"
                          className="wa-ghostSmall"
                          onClick={() => void loadAudioFromLibrary(item.id)}
                        >
                          Load
                        </button>
                      </div>
                      <div className="wa-historyMeta">
                        <span className="wa-tag">{formatDateTime(item.createdAt)}</span>
                        {item.durationSec ? <span className="wa-tag">{formatDuration(item.durationSec)}</span> : null}
                        <span className="wa-tag">{formatFileSize(item.sizeBytes)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </section>

        <div className="wa-panelStack">
          {status ? <div className="wa-status">{status}</div> : null}

          <section className="wa-card">
            <div className="wa-cardTitleRow">
              <div>
                <div className="wa-sectionKicker">02 Workflow</div>
                <div className="wa-cardTitle">Stages</div>
              </div>
            </div>

            <div className="wa-stageGrid">
              <section
                className={
                  expandedStage === 'transcribe' ? 'wa-stageCard wa-stageCardOpen' : 'wa-stageCard'
                }
              >
                <button
                  type="button"
                  className="wa-stageHeader"
                  onClick={() => setExpandedStage('transcribe')}
                >
                  <span className="wa-stageIndex">01</span>
                  <span className="wa-stageHeaderMain">
                    <span className="wa-stageTitle">Transcribe</span>
                  </span>
                </button>

                {expandedStage === 'transcribe' ? (
                  <div className="wa-stageBody">
                    <div className="wa-row">
                      <label className="wa-field">
                        <div className="wa-label">Model</div>
                        <select
                          className="wa-select"
                          value={selectedModel}
                          onChange={(e) => setSelectedModel(e.target.value)}
                          disabled={isBusy}
                        >
                          {MODEL_OPTIONS.map((o) => (
                            <option key={o.id} value={o.id}>
                              {o.label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="wa-field">
                        <div className="wa-label">Hint</div>
                        <input
                          className="wa-input"
                          value={languageHint}
                          onChange={(e) => setLanguageHint(e.target.value)}
                          placeholder="en"
                          disabled={isBusy}
                        />
                      </label>
                    </div>

                    <div className="wa-row">
                      <button
                        type="button"
                        className="wa-primary"
                        onClick={() => void transcribeCurrentAudio()}
                        disabled={!audioUrl || isBusy}
                      >
                        {isTranscribing ? 'Transcribing...' : 'Transcribe'}
                      </button>
                      <button
                        type="button"
                        className="wa-ghost"
                        onClick={() => void copyText(rawTranscript, 'transcription')}
                        disabled={!rawTranscript.trim()}
                      >
                        Copy
                      </button>
                    </div>

                    <label className="wa-textareaLabel">
                      <div className="wa-label">Output</div>
                      <textarea
                        className="wa-textarea wa-textareaSection"
                        value={rawTranscript}
                        readOnly
                        placeholder="Raw transcript"
                      />
                    </label>
                  </div>
                ) : null}
              </section>

              <section
                className={
                  expandedStage === 'refine' ? 'wa-stageCard wa-stageCardOpen' : 'wa-stageCard'
                }
              >
                <button
                  type="button"
                  className="wa-stageHeader"
                  onClick={() => setExpandedStage('refine')}
                >
                  <span className="wa-stageIndex">02</span>
                  <span className="wa-stageHeaderMain">
                    <span className="wa-stageTitle">Refine</span>
                  </span>
                </button>

                {expandedStage === 'refine' ? (
                  <div className="wa-stageBody">
                    <label className="wa-textareaLabel">
                      <div className="wa-label">Context</div>
                      <textarea
                        className="wa-textarea wa-textareaContext"
                        value={contextNotes}
                        onChange={(e) => setContextNotes(e.target.value)}
                        placeholder="Names, spellings, terms"
                        disabled={isBusy}
                      />
                    </label>

                    <div className="wa-row">
                      <button
                        type="button"
                        className="wa-primary"
                        onClick={() => void refineTranscript()}
                        disabled={!rawTranscript.trim() || isBusy}
                      >
                        {isRefining ? 'Refining...' : 'Refine'}
                      </button>
                      <button
                        type="button"
                        className="wa-primary wa-primaryAlt"
                        onClick={saveRefinement}
                        disabled={!correctedTranscript.trim() || !audioUrl || isBusy}
                      >
                        Save
                      </button>
                      <button
                        type="button"
                        className="wa-ghost"
                        onClick={() => void copyText(correctedTranscript, 'refined text')}
                        disabled={!correctedTranscript.trim()}
                      >
                        Copy
                      </button>
                    </div>

                    <label className="wa-textareaLabel">
                      <div className="wa-label">Output</div>
                      <textarea
                        className="wa-textarea wa-textareaSection"
                        value={correctedTranscript}
                        readOnly
                        placeholder="Clean English"
                      />
                    </label>
                  </div>
                ) : null}
              </section>

              <section
                className={
                  expandedStage === 'translate'
                    ? 'wa-stageCard wa-stageCardOpen'
                    : 'wa-stageCard'
                }
              >
                <button
                  type="button"
                  className="wa-stageHeader"
                  onClick={() => setExpandedStage('translate')}
                >
                  <span className="wa-stageIndex">03</span>
                  <span className="wa-stageHeaderMain">
                    <span className="wa-stageTitle">Translate</span>
                  </span>
                </button>

                {expandedStage === 'translate' ? (
                  <div className="wa-stageBody">
                    <label className="wa-field">
                      <div className="wa-label">Target</div>
                      <select
                        className="wa-select"
                        value={translationTarget}
                        onChange={(e) => setTranslationTarget(e.target.value as TranslationTarget)}
                        disabled={isBusy}
                      >
                        <option value="ja">Japanese</option>
                      </select>
                    </label>

                    <div className="wa-row">
                      <button
                        type="button"
                        className="wa-primary"
                        onClick={() => void translateTranscript()}
                        disabled={!correctedTranscript.trim() || isBusy}
                      >
                        {isTranslating ? 'Translating...' : 'Translate'}
                      </button>
                      <button
                        type="button"
                        className="wa-primary wa-primaryAlt"
                        onClick={saveTranslation}
                        disabled={!translatedJa.trim() || !audioUrl || isBusy}
                      >
                        Save
                      </button>
                      <button
                        type="button"
                        className="wa-ghost"
                        onClick={() => void copyText(translatedJa, 'translation')}
                        disabled={!translatedJa.trim()}
                      >
                        Copy
                      </button>
                    </div>

                    <label className="wa-textareaLabel">
                      <div className="wa-label">Output</div>
                      <textarea
                        className="wa-textarea wa-textareaSection"
                        value={translatedJa}
                        readOnly
                        placeholder="Japanese text"
                      />
                    </label>
                  </div>
                ) : null}
              </section>
            </div>
          </section>
        </div>

        <section className="wa-card wa-cardWide">
          <div className="wa-cardTitleRow">
            <div>
              <div className="wa-sectionKicker">03 Saved</div>
              <div className="wa-cardTitle">Records</div>
            </div>
            <div className="wa-muted">
              {records.length ? `${records.length} records` : 'Nothing saved yet.'}
            </div>
          </div>

          <div className="wa-recordsLayout">
            <div className="wa-subCard">
              <div className="wa-subCardTitle">List</div>
              {records.length === 0 ? (
                <div className="wa-emptySmall">Saved text appears here.</div>
              ) : (
                <div className="wa-historyList">
                  {records.map((r) => (
                    <div
                      key={r.id}
                      className={
                        r.id === selectedRecordId
                          ? 'wa-historyItem wa-historyItemActive'
                          : 'wa-historyItem'
                      }
                    >
                      <div className="wa-historyTop">
                        <button
                          type="button"
                          className="wa-recordMain"
                          onClick={() => loadRecord(r, 'raw')}
                        >
                          <span className="wa-historyName" title={r.audioName}>
                            {r.audioName}
                          </span>
                        </button>
                        <button
                          type="button"
                          className="wa-historyDel"
                          onClick={() => deleteRecord(r.id)}
                          aria-label={`Delete ${r.audioName}`}
                        >
                          x
                        </button>
                      </div>
                      <div className="wa-historyMeta">
                        <span className="wa-tag">{formatDateTime(r.createdAt)}</span>
                        {r.durationSec ? <span className="wa-tag">{formatDuration(r.durationSec)}</span> : null}
                        <span className="wa-tag">{shortModelName(r.model)}</span>
                        {r.correctedTranscript ? (
                          <button
                            type="button"
                            className="wa-tagBtn"
                            onClick={() => loadRecord(r, 'corrected')}
                          >
                            refined
                          </button>
                        ) : null}
                        {r.translatedJa ? (
                          <button
                            type="button"
                            className="wa-tagBtn"
                            onClick={() => loadRecord(r, 'ja')}
                          >
                            translated
                          </button>
                        ) : null}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="wa-subCard">
              <div className="wa-subCardTitle">Viewer</div>
              {!selectedRecord ? (
                <div className="wa-emptySmall">Select a record.</div>
              ) : (
                <div className="wa-stack">
                  <div className="wa-row">
                    <button
                      type="button"
                      className={selectedSavedView === 'raw' ? 'wa-chip wa-chipActive' : 'wa-chip'}
                      onClick={() => setSelectedSavedView('raw')}
                    >
                      Transcription
                    </button>
                    {selectedRecord.correctedTranscript ? (
                      <button
                        type="button"
                        className={
                          selectedSavedView === 'corrected' ? 'wa-chip wa-chipActive' : 'wa-chip'
                        }
                        onClick={() => setSelectedSavedView('corrected')}
                      >
                        Context refinement
                      </button>
                    ) : null}
                    {selectedRecord.translatedJa ? (
                      <button
                        type="button"
                        className={selectedSavedView === 'ja' ? 'wa-chip wa-chipActive' : 'wa-chip'}
                        onClick={() => setSelectedSavedView('ja')}
                      >
                        Translation
                      </button>
                    ) : null}
                    <button
                      type="button"
                      className="wa-copyIconBtn"
                      onClick={() => void copyText(selectedSavedText, 'saved text')}
                      disabled={!selectedSavedText.trim()}
                      aria-label="Copy text"
                      title="Copy text"
                    >
                      ⧉
                    </button>
                  </div>

                  <textarea
                    className="wa-textarea wa-textareaSection"
                    readOnly
                    value={selectedSavedText}
                  />
                </div>
              )}
            </div>
          </div>
        </section>
      </main>

      <footer className="wa-footer" />
    </div>
  )
}
