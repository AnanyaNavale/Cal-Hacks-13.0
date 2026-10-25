let lastCue = ''
let lastCueAt = 0
let ticket = 0
let context: AudioContext | null = null
let playing: AudioBufferSourceNode | null = null

export function unlockVoice() {
  const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
  if (!AudioCtx) return
  if (!context) context = new AudioCtx()
  if (context.state === 'suspended') void context.resume()
  const buffer = context.createBuffer(1, 1, context.sampleRate)
  const source = context.createBufferSource()
  source.buffer = buffer
  source.connect(context.destination)
  source.start()
}

export async function speak(text: string) {
  const now = Date.now()
  if (text === lastCue && now - lastCueAt < 2500) return true
  const mine = ++ticket
  try {
    const response = await fetch('/api/speak', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    })
    if (mine !== ticket) return true
    if (!response.ok) return false
    const bytes = await response.arrayBuffer()
    if (mine !== ticket) return true
    if (!context) unlockVoice()
    if (!context || context.state === 'suspended') await context?.resume()
    if (mine !== ticket) return true
    if (!context || context.state !== 'running') return false
    const audio = await context.decodeAudioData(bytes.slice(0))
    if (mine !== ticket) return true
    playing?.stop()
    const source = context.createBufferSource()
    source.buffer = audio
    source.connect(context.destination)
    source.start()
    playing = source
    lastCue = text
    lastCueAt = Date.now()
    return true
  } catch {
    return false
  }
}
