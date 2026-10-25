import { useEffect, useRef, useState } from 'react'
import { ConvexProvider, useMutation, useQuery } from 'convex/react'
import { api, convex } from './convexClient'
import { speak, unlockVoice } from './voice'

type Units = 'imperial' | 'metric'

type Body = {
  heightCm: number
  bustCm: number
  waistCm: number
  hipsCm: number
  inseamCm: number | null
  turnDone: boolean
}

type SizeRow = {
  label: string
  bustMin: number | null
  bustMax: number | null
  waistMin: number | null
  waistMax: number | null
  hipsMin: number | null
  hipsMax: number | null
}

type Product = {
  id: string
  name: string
  brand: string
  url: string
  image: string
  price: string
  areas: string[]
  fitNote: string
  inseamIn: number | null
  rows: SizeRow[]
}

type Zone = 'bust' | 'waist' | 'hips'

const CM = 2.54

function inches(cm: number) {
  return Math.round((cm / CM) * 10) / 10
}

function lengthFromCm(cm: number, units: Units) {
  return units === 'metric' ? `${Math.round(cm)} cm` : `${inches(cm)} in`
}

function heightFromCm(cm: number, units: Units) {
  if (units === 'metric') return `${Math.round(cm)} cm`
  const total = Math.round(cm / CM)
  return `${Math.floor(total / 12)} ft ${total % 12} in`
}

function measuresFor(units: Units, body: Body) {
  return {
    height: heightFromCm(body.heightCm, units),
    bust: lengthFromCm(body.bustCm, units),
    waist: lengthFromCm(body.waistCm, units),
    hips: lengthFromCm(body.hipsCm, units),
    inseam: body.inseamCm == null ? 'Not captured' : lengthFromCm(body.inseamCm, units),
  }
}

type Measures = ReturnType<typeof measuresFor>

function rangeFor(row: SizeRow, zone: Zone): [number, number] | null {
  const min = zone === 'bust' ? row.bustMin : zone === 'waist' ? row.waistMin : row.hipsMin
  const max = zone === 'bust' ? row.bustMax : zone === 'waist' ? row.waistMax : row.hipsMax
  if (min == null || max == null) return null
  return [min, max]
}

function recommend(product: Product, body: Body) {
  const values: Record<Zone, number> = { bust: inches(body.bustCm), waist: inches(body.waistCm), hips: inches(body.hipsCm) }
  const zones = product.areas.filter((area): area is Zone => area === 'bust' || area === 'waist' || area === 'hips')
  let best: { row: SizeRow; score: number; index: number } | null = null
  for (let index = 0; index < product.rows.length; index += 1) {
    const row = product.rows[index]
    let score = 0
    let used = 0
    for (const zone of zones) {
      const range = rangeFor(row, zone)
      if (!range) continue
      used += 1
      const value = values[zone]
      score += value < range[0] ? range[0] - value : value > range[1] ? value - range[1] : 0
    }
    if (!used) continue
    if (!best || score < best.score - 0.05 || (Math.abs(score - best.score) < 0.05 && index > best.index)) best = { row, score, index }
  }
  if (!best) return { label: '—', badge: 'No chart', note: 'The saved chart has no measurements for the areas this garment covers.' }
  const chosen = best
  const parts = zones.flatMap(zone => {
    const range = rangeFor(chosen.row, zone)
    if (!range) return []
    const shown = range[0] === range[1] ? `${range[0]} in` : `${range[0]}–${range[1]} in`
    return [`${zone} ${values[zone]} in sits against ${shown}`]
  })
  let note = `${parts.join('; ')}.`
  if (product.id === 'jeans') {
    const heightIn = body.heightCm / CM
    const length = heightIn <= 61 ? 28 : heightIn <= 63 ? 30 : heightIn <= 67 ? 32 : 34
    note += ` Levi's length chart puts this height at a ${length} in inseam.`
    if (body.inseamCm != null) note += ` The scan measured ${inches(body.inseamCm)} in.`
  }
  if (product.inseamIn != null && body.inseamCm != null && product.id !== 'jeans') {
    const mine = inches(body.inseamCm)
    const gap = Math.round((product.inseamIn - mine) * 10) / 10
    note += gap === 0
      ? ` This pair is sold with a ${product.inseamIn} in inseam, the same as the scan.`
      : ` This pair is sold with a ${product.inseamIn} in inseam, ${Math.abs(gap)} in ${gap > 0 ? 'longer' : 'shorter'} than the scan.`
  }
  if (!body.turnDone) note += ' The turn was skipped, so circumferences use the scanner’s estimated depth.'
  return { label: chosen.row.label, badge: `Size ${chosen.row.label}`, note }
}

type Step = 'welcome' | 'scan' | 'review' | 'import' | 'compare' | 'questions'

const steps: { id: Step; label: string }[] = [
  { id: 'scan', label: 'Scan' },
  { id: 'review', label: 'Review' },
  { id: 'import', label: 'Garment' },
  { id: 'compare', label: 'Compare' },
  { id: 'questions', label: 'Ask' },
]

const Icon = ({ name, className = '' }: { name: 'arrow' | 'camera' | 'lock' | 'spark' | 'eye' | 'link' | 'check' | 'chevron'; className?: string }) => {
  const paths = {
    arrow: <><path d="M5 12h14M13 6l6 6-6 6"/></>,
    camera: <><path d="M15 8h.01"/><path d="M4 7h3l2-3h6l2 3h3v12H4z"/><circle cx="12" cy="13" r="3"/></>,
    lock: <><rect x="5" y="10" width="14" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></>,
    spark: <><path d="m12 3 1.3 4.2L17 9l-3.7 1.8L12 15l-1.3-4.2L7 9l3.7-1.8z"/><path d="m5 15 .7 2.3L8 18l-2.3.7L5 21l-.7-2.3L2 18l2.3-.7z"/></>,
    eye: <><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12Z"/><circle cx="12" cy="12" r="2.5"/></>,
    link: <><path d="M10 13a5 5 0 0 0 7.5.5l2-2a5 5 0 0 0-7-7l-1.2 1.2"/><path d="M14 11a5 5 0 0 0-7.5-.5l-2 2a5 5 0 0 0 7 7l1.2-1.2"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    chevron: <path d="m9 18 6-6-6-6"/>,
  }
  return <svg aria-hidden="true" width="19" height="19" className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>
}

function Button({ children, onClick, secondary = false, disabled = false, className = '', type = 'button' }: {
  children: React.ReactNode; onClick?: () => void; secondary?: boolean; disabled?: boolean; className?: string; type?: 'button' | 'submit'
}) {
  return <button type={type} disabled={disabled} onClick={onClick} className={`btn ${secondary ? 'btn-secondary' : 'btn-primary'} ${className}`}>{children}</button>
}

function UnitSwitch({ units, onChange, className = '' }: { units: Units; onChange: (units: Units) => void; className?: string }) {
  return <div className={`unit-switch ${className}`} role="group" aria-label="Measurement units">
    <button type="button" aria-pressed={units === 'imperial'} onClick={() => onChange('imperial')}>ft / in</button>
    <button type="button" aria-pressed={units === 'metric'} onClick={() => onChange('metric')}>cm</button>
  </div>
}

function Silhouette({ heatmap = false, onArea, marked = false, measures, areas = [] }: { heatmap?: boolean; onArea?: (area: string) => void; marked?: boolean; measures?: Measures; areas?: string[] }) {
  return (
    <div className="silhouette-wrap" aria-label={heatmap ? 'Interactive body fit heatmap' : 'Body silhouette'}>
      <svg viewBox="0 0 240 420" role="img" aria-hidden={heatmap}>
        <ellipse cx="120" cy="47" rx="30" ry="35" className="body-fill" />
        <path className="body-fill" d="M83 83c-15 11-23 39-19 71l8 73-17 151c-2 19 28 24 34 4l26-121h10l26 121c6 20 36 15 34-4l-17-151 8-73c4-32-4-60-19-71-18 9-56 9-74 0Z"/>
      </svg>
      {heatmap && areas.map(area => <button className={`area-dot ${area === 'hips' ? 'hip' : area}`} aria-label={`Ask about ${area} fit`} key={area} onClick={() => onArea?.(area)} />)}
      {marked && <>
        <span className="body-mark bust">{measures?.bust}</span>
        <span className="body-mark waist">{measures?.waist}</span>
        <span className="body-mark hip">{measures?.hips}</span>
        <span className="body-mark inseam">{measures?.inseam}</span>
      </>}
    </div>
  )
}

function Header({ step, go, units, onUnits }: { step: Step; go: (step: Step) => void; units: Units; onUnits: (units: Units) => void }) {
  const active = Math.max(0, steps.findIndex(s => s.id === step))
  return (
    <header className="site-header">
      <button className="brand" onClick={() => go('welcome')} aria-label="Fit home">fit<span>.</span></button>
      {step !== 'welcome' && <nav aria-label="Fit journey" className="steps">
        {steps.map((item, i) => <button key={item.id} onClick={() => i <= active && go(item.id)} aria-current={item.id === step ? 'step' : undefined} className={i <= active ? 'active' : ''}>
          <span>{i < active ? <Icon name="check" /> : i + 1}</span><em>{item.label}</em>
        </button>)}
      </nav>}
      <div className="header-tools"><UnitSwitch units={units} onChange={onUnits} /><div className="privacy"><Icon name="lock" />Private by design</div></div>
    </header>
  )
}

function Welcome({ go }: { go: (s: Step) => void }) {
  return <main className="welcome">
    <section className="hero-copy">
      <div className="eyebrow"><Icon name="spark" /> Fit, without the fitting room</div>
      <h1>Less uncertainty.<br />More confident choices.</h1>
      <p>See how clothes may fit your unique shape, from the comfort of home. No judgment, no standard bodies—just useful guidance.</p>
      <div className="hero-actions">
        <Button onClick={() => go('scan')}>Start body scan <Icon name="arrow" /></Button>
        <span>Front pose, then a slow turn</span>
      </div>
      <div className="trust-row">
        <div><Icon name="lock" /><span><b>Stays on your device</b><small>Frames are never uploaded</small></span></div>
        <div><Icon name="camera" /><span><b>You’re in control</b><small>Delete or retry anytime</small></span></div>
      </div>
      <div className="how-it-works" aria-label="How Fit works">
        <p><b>How Fit helps.</b> Fit is designed to turn your measurements into useful clothing recommendations.</p>
        <ol>
          <li><span>1</span><div><b>Scan your shape</b><small>Capture measurements privately</small></div></li>
          <li><span>2</span><div><b>Add any garment</b><small>Compare its cut and size chart</small></div></li>
          <li><span>3</span><div><b>Find your fit</b><small>Get a size recommendation with a reason</small></div></li>
        </ol>
      </div>
    </section>
    <section className="hero-art" aria-label="A diverse group of body silhouettes">
      <div className="arch">
        <span className="shape shape-1" />
        <span className="shape shape-2" />
        <span className="shape shape-3" />
        <span className="shape shape-4" />
      </div>
      <div className="floating-card"><span><Icon name="check" /></span><div><small>Made for every body</small><b>Body-neutral guidance</b></div></div>
    </section>
  </main>
}

const DONE_CUE = 'All done. Your measurements are on screen.'
const OPENING_CUE = 'Step 1. Take your shoes off, and stand about nine feet back, so your whole body, including your feet, is in view. Hold your arms out in an A shape, with your feet apart.'

type ScanCue = { step: string; title: string; say: string; hint: string; progress: number | null }

function narration(cue: ScanCue, memory: { step: string; hint: string; hintSince: number; spoken: string }) {
  const hint = cue.hint.replace(/\s*\(press N to skip\)/gi, '').trim()
  if (cue.step !== memory.step) {
    memory.step = cue.step
    memory.hint = hint
    memory.hintSince = Date.now()
    memory.spoken = cue.say
    return cue.say
  }
  if (hint !== memory.hint) {
    memory.hint = hint
    memory.hintSince = Date.now()
    return null
  }
  if (!hint || hint === memory.spoken || cue.say.toLowerCase().includes(hint.toLowerCase())) return null
  if (/^\[n\]|^hold still$|^keep turning slowly$/i.test(hint)) return null
  const urgent = /perfect|get ready to turn|lift your arms|hands on your head|same spot|feet in view|camera moving/i.test(hint)
  if (Date.now() - memory.hintSince < (urgent ? 1200 : 6000)) return null
  memory.spoken = hint
  memory.hintSince = Date.now()
  return hint
}

type RawProfile = { heightCm: number; measurementsCm: { bust: number; waist: number; hips: number; inseam?: number }; turnDone: boolean }

function heightCmFromForm(units: Units, feet: string, inchesValue: string, cm: string) {
  if (units === 'metric') {
    const value = Number(cm)
    return cm !== '' && Number.isFinite(value) ? value : null
  }
  if (feet === '' || inchesValue === '' || !Number.isFinite(Number(feet)) || !Number.isFinite(Number(inchesValue))) return null
  return (Number(feet) * 12 + Number(inchesValue)) * CM
}

function Scan({ go, logHeight, onProfile, units, onUnits, feet, inches: inchesValue, cm, setFeet, setInches, setCm }: {
  go: (s: Step) => void
  logHeight: (height: string) => void
  onProfile: (profile: RawProfile) => void
  units: Units
  onUnits: (units: Units) => void
  feet: string
  inches: string
  cm: string
  setFeet: (value: string) => void
  setInches: (value: string) => void
  setCm: (value: string) => void
}) {
  const request = useRef(0)
  const voiceMemory = useRef({ step: '', hint: '', hintSince: 0, spoken: '' })
  const [phase, setPhase] = useState<'height' | 'running' | 'error'>('height')
  const [message, setMessage] = useState('')
  const [cue, setCue] = useState<ScanCue | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [voiceOn, setVoiceOn] = useState(true)
  const previewUrl = useRef<string | null>(null)
  const heightCm = heightCmFromForm(units, feet, inchesValue, cm)

  useEffect(() => () => { request.current += 1 }, [])

  const start = async () => {
    if (heightCm == null || heightCm < 100 || heightCm > 250) return
    const current = ++request.current
    voiceMemory.current = { step: 'front_wait', hint: '', hintSince: Date.now(), spoken: OPENING_CUE }
    logHeight(heightFromCm(heightCm, units))
    unlockVoice()
    void speak(OPENING_CUE).then(ok => { if (current === request.current) setVoiceOn(ok) })
    setPhase('running')
    setCue(null)
    setPreview(null)
    setMessage('Opening the scanner…')
    try {
      const response = await fetch('/api/scan', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ heightCm }) })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Could not open the scanner.')
      const poll = async () => {
        if (current !== request.current) return
        const status = await fetch('/api/scan')
        const state = await status.json()
        if (current !== request.current) return
        const nextCue = state.scan as ScanCue | null
        if (nextCue?.step) {
          setCue(nextCue)
          const line = narration(nextCue, voiceMemory.current)
          if (line) void speak(line).then(ok => { if (current === request.current) setVoiceOn(ok) })
        }
        if (state.status === 'running') {
          const frame = await fetch('/api/scan/frame')
          if (current === request.current && frame.status === 200) {
            const url = URL.createObjectURL(await frame.blob())
            if (previewUrl.current) URL.revokeObjectURL(previewUrl.current)
            previewUrl.current = url
            setPreview(url)
          }
        }
        if (state.status === 'complete') {
          if (voiceMemory.current.spoken !== DONE_CUE) void speak(DONE_CUE)
          onProfile(state.profile)
          go('review')
          return
        }
        if (state.status === 'error' || state.status === 'cancelled') {
          setPhase('error')
          setMessage(state.error || 'The scan stopped before it saved measurements.')
          return
        }
        window.setTimeout(() => { void poll() }, 400)
      }
      window.setTimeout(() => { void poll() }, 400)
    } catch (error) {
      if (current !== request.current) return
      setPhase('error')
      setMessage(error instanceof Error ? error.message : 'Could not open the scanner.')
    }
  }

  const cancel = async () => {
    request.current += 1
    await fetch('/api/scan/cancel', { method: 'POST' }).catch(() => {})
    setPhase('height')
    setMessage('')
  }

  const stepIndex = cue?.step === 'turn_wait' || cue?.step === 'turn' ? 2 : cue?.step === 'live' ? 3 : phase === 'running' ? 1 : 0
  return <main className="page scan-page">
    <div className="page-title"><span className="eyebrow">Guided body scan</span><h1>Let’s get your best fit</h1><p>Wear fitted clothing. The camera stays on this computer, and Cartesia talks you through each step.</p></div>
    <div className="scan-layout">
      <section className="camera-card">
        <div className="camera-meta"><span>{phase === 'height' ? 'Enter height' : cue?.title || 'Body scanner'}</span><small>{phase === 'height' ? '1 of 2' : '2 of 2'}</small></div>
        <div className="camera-view">
          {phase === 'height' && <div className="height-prompt"><div className="height-entry"><UnitSwitch units={units} onChange={onUnits} />{units === 'metric' ? <label>Your height<span><input aria-label="Height in centimetres" type="number" min="100" max="250" value={cm} onChange={e => setCm(e.target.value)} /><em>cm</em></span></label> : <div className="height-pair" role="group" aria-label="Your height"><label>Feet<span><input aria-label="Height in feet" type="number" min="3" max="8" value={feet} onChange={e => setFeet(e.target.value)} /><em>ft</em></span></label><label>Inches<span><input aria-label="Remaining inches" type="number" min="0" max="11" value={inchesValue} onChange={e => setInches(e.target.value)} /><em>in</em></span></label></div>}</div></div>}
          {phase === 'running' && <div className="scanner-stage">
            {preview && <img src={preview} alt="Live view from the body scanner" />}
            <div className="scanner-caption">
              <b>{cue?.title || 'Opening the scanner'}</b>
              <small>{cue?.hint || message}</small>
              {!voiceOn && <button className="hear-again" type="button" onClick={() => { unlockVoice(); void speak(voiceMemory.current.spoken || OPENING_CUE).then(setVoiceOn) }}>Click to hear Cartesia</button>}
              {cue?.progress != null && <div className="scan-progress" aria-hidden="true"><span style={{ width: `${Math.round(cue.progress * 100)}%` }} /></div>}
            </div>
          </div>}
          {phase === 'error' && <div className="camera-message error"><h2>The scanner did not finish</h2><p>{message}</p></div>}
        </div>
        <div className="capture-bar">
          {phase === 'height' && <Button disabled={heightCm == null || heightCm < 100 || heightCm > 250} onClick={() => { void start() }}>Open body scanner <Icon name="arrow" /></Button>}
          {phase === 'running' && <Button secondary onClick={() => { void cancel() }}>Cancel scan</Button>}
          {phase === 'error' && <Button onClick={() => { setPhase('height'); setMessage('') }}>Try again</Button>}
        </div>
      </section>
      <aside className="tips-card">
        <h2>{phase === 'running' ? 'Cartesia is guiding this scan' : 'Your height sets the scale'}</h2>
        <p>{phase === 'running' ? cue?.say || 'The scanner window still opens on this computer. Cartesia reads each step as the camera sees you.' : 'The scanner will not guess your height. It uses the number you enter to turn the camera view into inches.'}</p>
        <ol className="scan-checklist">
          {['Save your height', 'Hold the front pose', 'Turn slowly in a full circle', 'Come back here for the measurements'].map((item, index) => <li className={index < stepIndex ? 'complete' : index === stepIndex ? 'current' : ''} key={item}><span>{index < stepIndex ? <Icon name="check" /> : index + 1}</span><b>{item}</b></li>)}
        </ol>
      </aside>
    </div>
  </main>
}

function ConnectedScan(props: Omit<Parameters<typeof Scan>[0], 'logHeight'>) {
  const logScan = useMutation(api.scans.logScan)
  return <Scan {...props} logHeight={height => { void logScan({ height }).catch(() => {}) }} />
}

function Review({ go, measures, summary, turnDone }: { go: (s: Step) => void; measures: Measures; summary: string; turnDone: boolean }) {
  return <main className="page narrow-wide">
    <div className="page-title"><span className="eyebrow">Scan complete</span><h1>Your body profile</h1><p>These are webcam estimates from the Python scanner. They are not a tape measure.</p></div>
    <div className="review-grid">
      <section className="card silhouette-card"><Silhouette marked measures={measures} /><div className="confidence"><span><Icon name="check" /></span><div><small>Source</small><b>{turnDone ? 'Front and turn' : 'Front only'}</b></div></div></section>
      <section className="card measurements">
        <div className="card-head"><div><h2>Your measurements</h2></div></div>
        <p>{summary}</p>
        {([['Height', measures.height], ['Bust', measures.bust], ['Waist', measures.waist], ['Hips', measures.hips], ['Inseam', measures.inseam]] as const).map(([label, value]) => <div className="measure-row" key={label}><span>{label}</span><b>{value}</b></div>)}
        <div className="review-actions"><Button onClick={() => go('compare')}>See clothing matches <Icon name="arrow" /></Button><Button secondary onClick={() => go('scan')}>Scan again</Button></div>
      </section>
    </div>
  </main>
}

function ImportGarment({ go }: { go: (s: Step) => void }) {
  const [hasChart, setHasChart] = useState(true)
  return <main className="page narrow">
    <div className="page-title"><span className="eyebrow">Add a garment</span><h1>What are you considering?</h1><p>Paste a product link or add details yourself. We’ll use them only for this comparison.</p></div>
    <form className="card import-form" onSubmit={e => { e.preventDefault(); go('compare') }}>
      <label className="full">Product URL <span className="input-icon"><Icon name="link" /><input type="url" placeholder="https://shop.com/product" /></span></label>
      <div className="divider"><span>or add manually</span></div>
      <label>Garment name<input required placeholder="e.g. Linen wrap dress" /></label>
      <label>Fabric<select defaultValue="stretch"><option value="stretch">Stretch woven</option><option>Non-stretch woven</option><option>Knit</option></select></label>
      <label className="full">Product image<span className="upload"><Icon name="camera" /><span><b>Choose a product image</b><small>PNG or JPG, up to 8MB</small></span><input type="file" accept="image/*" /></span></label>
      <label className="full">Available sizes<input required placeholder="e.g. 14, 16, 18, 20, 22, 24" /></label>
      <label className="full">Size chart<textarea placeholder="Paste the brand’s size chart or key measurements" /></label>
      <label className="check-row full"><input type="checkbox" checked={!hasChart} onChange={e => setHasChart(!e.target.checked)} /><span><b>This product has no size chart</b><small>We can still offer general fabric and cut guidance.</small></span></label>
      {!hasChart && <div className="missing full"><b>Limited comparison</b><span>Without a size chart, we won’t recommend a specific size.</span></div>}
      <Button type="submit" className="full">Compare fit <Icon name="arrow" /></Button>
    </form>
  </main>
}

function Compare({ go, products, body, measures, onAsk }: { go: (s: Step) => void; products: Product[]; body: Body; measures: Measures; onAsk: (id: string) => void }) {
  const [selected, setSelected] = useState(products[0]?.id ?? '')
  const item = products.find(entry => entry.id === selected) ?? products[0]
  if (!item) return <main className="page"><div className="page-title"><h1>No saved clothes yet</h1><p>The product table is empty, so there is nothing to size.</p></div></main>
  const fit = recommend(item, body)
  return <main className="page compare-page">
    <div className="page-title split-title"><div><span className="eyebrow">Fit comparison</span><h1>Matches for your scan</h1></div><Button secondary onClick={() => go('review')}>Back to measurements</Button></div>
    <div className="compare-grid">
      <section className="garment-grid" aria-label="Clothing matches">
        {products.map(entry => {
          const entryFit = recommend(entry, body)
          return <button className={entry.id === item.id ? 'selected' : ''} key={entry.id} onClick={() => setSelected(entry.id)} aria-pressed={entry.id === item.id}>
            {entry.image ? <img src={entry.image} alt="" /> : <span className="garment-fallback">{entry.brand}</span>}
            <b>{entry.name}</b>
            <small>{entryFit.badge}</small>
          </button>
        })}
      </section>
      <section className="recommendation">
        <div className="viewer">{item.image ? <img src={item.image} alt={item.name} /> : <span className="garment-fallback">{item.brand}</span>}</div>
        <div className="rec-top"><span><Icon name="spark" /></span><div><small>Recommended</small><h2>{fit.label}</h2></div></div>
        <h3>{item.name}</h3>
        <p>{fit.note}</p>
        {item.fitNote && <p>{item.fitNote}</p>}
        <div className="fit-note"><b>{fit.badge}{item.price ? ` · ${item.price}` : ''}</b><span>From a {measures.height} scan: bust {measures.bust}, waist {measures.waist}, hips {measures.hips}, inseam {measures.inseam}.</span></div>
        <a className="product-link" href={item.url} target="_blank" rel="noreferrer">View this {item.brand} page</a>
        <Button onClick={() => onAsk(item.id)}>Ask about this fit <Icon name="arrow" /></Button>
      </section>
    </div>
  </main>
}

function Questions({ go, garmentId, products, body, measures }: { go: (s: Step) => void; garmentId: string; products: Product[]; body: Body; measures: Measures }) {
  const item = products.find(entry => entry.id === garmentId) ?? products[0]
  const fit = item ? recommend(item, body) : { label: '—', badge: 'No chart', note: '' }
  const [area, setArea] = useState(item?.areas[0] ?? 'bust')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [asking, setAsking] = useState(false)
  const [error, setError] = useState('')
  if (!item) return <main className="page"><div className="page-title"><h1>No saved clothes yet</h1></div></main>
  const ask = async () => {
    setAsking(true)
    setError('')
    setAnswer('')
    try {
      const wearer = `${measures.height}, bust ${measures.bust}, waist ${measures.waist}, hips ${measures.hips}, inseam ${measures.inseam}`
      const response = await fetch('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ garment: item.name, size: fit.label, area, covers: item.areas, wearer, question }) })
      const data = await response.json()
      if (!response.ok) setError(data.error || 'Could not get guidance.')
      else setAnswer(data.answer)
    } catch {
      setError('Could not reach the guidance service.')
    } finally {
      setAsking(false)
    }
  }
  const prompts = [
    `Will the ${item.name} feel tight at my ${area}?`,
    `Does size ${fit.label} leave enough room at my ${area}?`,
    `How will the ${item.name} move at my ${area}?`,
  ]
  return <main className="page questions-page">
    <div className="page-title"><span className="eyebrow">{item.name}</span><h1>Where would you like more detail?</h1><p>The markers are only on the parts of the body this piece covers.</p></div>
    <div className="question-grid">
      <section className="card question-body"><Silhouette heatmap areas={item.areas} onArea={setArea} /><p>Tap a marker this garment touches</p></section>
      <section className="card ask-card">
        <span className="selected-area">Selected area <b>{area}</b></span>
        <h2>What would you like to know about the {item.name}?</h2>
        <div className="suggestions">{prompts.map(q => <button key={q} onClick={() => setQuestion(q)}>{q}<Icon name="chevron" /></button>)}</div>
        <label>Your question<textarea value={question} onChange={e => setQuestion(e.target.value)} placeholder={`Ask about the ${item.name}…`} /></label>
        <Button disabled={!question || asking} onClick={() => { void ask() }}>{asking ? 'Asking…' : 'Get fit guidance'} <Icon name="arrow" /></Button>
        {answer && <p className="answer">{answer}</p>}
        {error && <p className="demo-note">{error}</p>}
        <button className="text-button" onClick={() => go('compare')}>Back to fit comparison</button>
      </section>
    </div>
  </main>
}

function ConvexCatalog({ onProducts }: { onProducts: (products: Product[]) => void }) {
  const products = useQuery(api.catalog.list) as Product[] | undefined
  useEffect(() => { if (products) onProducts(products) }, [products, onProducts])
  return null
}

function AppShell() {
  const [step, setStep] = useState<Step>('welcome')
  const [garmentId, setGarmentId] = useState('')
  const [units, setUnits] = useState<Units>('imperial')
  const [feet, setFeet] = useState('5')
  const [inchesValue, setInches] = useState('7')
  const [cm, setCm] = useState('170')
  const [body, setBody] = useState<Body | null>(null)
  const [products, setProducts] = useState<Product[]>([])
  useEffect(() => {
    if (convex) return
    void fetch('/api/catalog').then(response => response.json()).then(data => setProducts(data.products || [])).catch(() => setProducts([]))
  }, [])
  const changeUnits = (next: Units) => {
    if (next === units) return
    if (next === 'metric') {
      const total = Number(feet) * 12 + Number(inchesValue)
      if (feet !== '' && inchesValue !== '' && Number.isFinite(total)) setCm(String(Math.round(total * CM)))
    } else {
      const totalCm = Number(cm)
      if (cm !== '' && Number.isFinite(totalCm)) {
        const total = Math.round(totalCm / CM)
        setFeet(String(Math.floor(total / 12)))
        setInches(String(total % 12))
      }
    }
    setUnits(next)
  }
  const acceptProfile = (profile: RawProfile) => {
    setBody({
      heightCm: profile.heightCm,
      bustCm: profile.measurementsCm.bust,
      waistCm: profile.measurementsCm.waist,
      hipsCm: profile.measurementsCm.hips,
      inseamCm: profile.measurementsCm.inseam ?? null,
      turnDone: Boolean(profile.turnDone),
    })
  }
  const measures = body ? measuresFor(units, body) : null
  const summary = body && measures
    ? products.map(product => `${product.name}: ${recommend(product, body).badge}`).join(' ') || 'Measurements are ready. Clothing charts will appear when the catalog table has rows.'
    : ''
  const go = (next: Step) => { setStep(next); window.scrollTo({ top: 0, behavior: 'smooth' }) }
  const scanProps = { go, onProfile: acceptProfile, units, onUnits: changeUnits, feet, inches: inchesValue, cm, setFeet, setInches, setCm }
  return <div className="app">{convex && <ConvexCatalog onProducts={setProducts} />}<Header step={step} go={go} units={units} onUnits={changeUnits} />{step === 'welcome' && <Welcome go={go} />}{step === 'scan' && (convex ? <ConnectedScan {...scanProps} /> : <Scan {...scanProps} logHeight={() => {}} />)}{step === 'review' && body && measures && <Review go={go} measures={measures} summary={summary} turnDone={body.turnDone} />}{step === 'import' && <ImportGarment go={go} />}{step === 'compare' && body && measures && <Compare go={go} products={products} body={body} measures={measures} onAsk={id => { setGarmentId(id); go('questions') }} />}{step === 'questions' && body && measures && <Questions go={go} garmentId={garmentId} products={products} body={body} measures={measures} />}<footer>Fit guidance, not a promise. Webcam estimates and saved size charts can both be off.</footer></div>
}

export default function App() {
  if (!convex) return <AppShell />
  return <ConvexProvider client={convex}><AppShell /></ConvexProvider>
}
