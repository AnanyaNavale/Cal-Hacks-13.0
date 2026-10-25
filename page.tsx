"use client";

import { usePoseTracking } from "./usePoseTracking";
import { useEffect, useRef, useState } from 'react'

type Step = 'welcome' | 'scan' | 'review' | 'import' | 'compare' | 'questions'
type CameraState = 'idle' | 'loading' | 'ready' | 'denied' | 'error'

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
  return <svg aria-hidden="true" className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>
}

function Button({ children, onClick, secondary = false, disabled = false, className = '', type = 'button' }: {
  children: React.ReactNode; onClick?: () => void; secondary?: boolean; disabled?: boolean; className?: string; type?: 'button' | 'submit'
}) {
  return <button type={type} disabled={disabled} onClick={onClick} className={`btn ${secondary ? 'btn-secondary' : 'btn-primary'} ${className}`}>{children}</button>
}

function Silhouette({ heatmap = false, onArea }: { heatmap?: boolean; onArea?: (area: string) => void }) {
  return (
    <div className="silhouette-wrap" aria-label={heatmap ? 'Interactive body fit heatmap' : 'Body silhouette'}>
      <svg viewBox="0 0 240 420" role="img" aria-hidden={heatmap}>
        <ellipse cx="120" cy="47" rx="30" ry="35" className={heatmap ? 'zone-comfort' : 'body-fill'} />
        <path className={heatmap ? 'zone-snug' : 'body-fill'} d="M83 83c-15 11-23 39-19 71l8 73-17 151c-2 19 28 24 34 4l26-121h10l26 121c6 20 36 15 34-4l-17-151 8-73c4-32-4-60-19-71-18 9-56 9-74 0Z"/>
        {heatmap && <>
          <path className="zone-tight" d="M69 145c10 17 92 17 102 0l-3 47c-20 11-76 11-96 0Z"/>
          <path className="zone-loose" d="m72 205 96 0 7 56H65Z"/>
        </>}
      </svg>
      {heatmap && <>
        <button className="area-dot bust" aria-label="Ask about bust fit" onClick={() => onArea?.('bust')} />
        <button className="area-dot waist" aria-label="Ask about waist fit" onClick={() => onArea?.('waist')} />
        <button className="area-dot hip" aria-label="Ask about hip fit" onClick={() => onArea?.('hips')} />
      </>}
    </div>
  )
}

function Header({ step, go }: { step: Step; go: (step: Step) => void }) {
  const active = Math.max(0, steps.findIndex(s => s.id === step))
  return (
    <header className="site-header">
      <button className="brand" onClick={() => go('welcome')} aria-label="Fit home">fit<span>.</span></button>
      {step !== 'welcome' && <nav aria-label="Fit journey" className="steps">
        {steps.map((item, i) => <button key={item.id} onClick={() => i <= active && go(item.id)} aria-current={item.id === step ? 'step' : undefined} className={i <= active ? 'active' : ''}>
          <span>{i < active ? <Icon name="check" /> : i + 1}</span><em>{item.label}</em>
        </button>)}
      </nav>}
      <div className="privacy"><Icon name="lock" />Private by design</div>
    </header>
  )
}

function Welcome({ go }: { go: (s: Step) => void }) {
  return <main className="welcome">
    <section className="hero-copy">
      <div className="eyebrow"><Icon name="spark" /> Fit, without the fitting room</div>
      <h1>Find your fit<br />before you buy.</h1>
      <p>See how clothes may fit your unique shape, from the comfort of home. No judgment, no standard bodies—just useful guidance.</p>
      <div className="hero-actions">
        <Button onClick={() => go('scan')}>Start body scan <Icon name="arrow" /></Button>
        <span>Takes about 2 minutes</span>
      </div>
      <div className="trust-row">
        <div><Icon name="lock" /><span><b>Stays on your device</b><small>Frames are never uploaded</small></span></div>
        <div><Icon name="camera" /><span><b>You’re in control</b><small>Delete or retry anytime</small></span></div>
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

function Scan({ go }: { go: (s: Step) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const cameraRequest = useRef(0)
  const [camera, setCamera] = useState<CameraState>('idle')
  const [stage, setStage] = useState(0)
  const [height, setHeight] = useState('')
  const [manual, setManual] = useState(false)
  const trackingMessage = usePoseTracking(camera === 'ready' && !manual, videoRef, canvasRef)
  const stopCamera = () => {
    cameraRequest.current += 1
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setCamera('idle')
  }
  const scanStages = [
    { title: 'Centre your face', short: 'Face calibration', detail: 'Move closer and place your face inside the oval. Keep a neutral expression and look straight ahead.', action: 'Continue (calibration not connected)' },
    { title: 'Add your height', short: 'Enter height', detail: 'Your height provides the scale needed to estimate proportions. We will never guess it from the camera.', action: 'Save height & continue' },
    { title: 'Show your full body', short: 'Full-body framing', detail: 'Step back until your entire body—from the top of your head to the soles of your shoes—is visible.', action: 'Confirm full body is visible' },
    { title: 'Raise your arms into an A', short: 'A-pose', detail: 'Face forward and hold both arms diagonally down and away from your body, about 45 degrees. Keep your feet visible.', action: 'Continue with guide' },
    { title: 'Turn sideways, arms down', short: 'Side turn', detail: 'Lower your arms to your sides, then turn 90 degrees to your left while keeping your whole body in frame.', action: 'Explore demo results' },
  ]
  const currentStage = scanStages[stage]

  const startCamera = async () => {
    const request = ++cameraRequest.current
    streamRef.current?.getTracks().forEach(t => t.stop())
    setCamera('loading')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
      if (request !== cameraRequest.current) { stream.getTracks().forEach(t => t.stop()); return }
      streamRef.current = stream
      if (videoRef.current) videoRef.current.srcObject = stream
      setCamera('ready')
    } catch (error) {
      if (request !== cameraRequest.current) return
      setCamera(error instanceof DOMException && error.name === 'NotAllowedError' ? 'denied' : 'error')
    }
  }
  useEffect(() => () => { cameraRequest.current += 1; streamRef.current?.getTracks().forEach(track => track.stop()) }, [])
  const nextCapture = () => {
    if (stage === 1 && !height) return
    if (stage < scanStages.length - 1) setStage(stage + 1)
    else { stopCamera(); go('review') }
  }

  if (manual) return <main className="page narrow">
    <div className="page-title"><span className="eyebrow">Manual fallback</span><h1>Enter what you know</h1><p>This form is a visual preview. Entries are not yet connected to a body profile.</p></div>
    <form className="card form-grid" onSubmit={e => { e.preventDefault(); go('review') }}>
      {['Height', 'Bust', 'Waist', 'Hips'].map((label, i) => <label key={label}>{label}<span><input required type="number" min="1" placeholder={i ? '—' : '165'} /><em>cm</em></span></label>)}
      <Button type="submit" className="full">Review my details <Icon name="arrow" /></Button>
      <Button secondary onClick={() => setManual(false)} className="full">Back to camera</Button>
    </form>
  </main>

  return <main className="page scan-page">
    <div className="page-title"><span className="eyebrow">Guided body scan</span><h1>Let’s get your best fit</h1><p>Wear fitted clothing and follow the guide. Nothing is recorded or uploaded.</p></div>
    <div className="scan-layout">
      <section className="camera-card">
        <div className="camera-meta"><span>{currentStage.short}</span><small>{stage + 1} of {scanStages.length}</small></div>
        <div className={`camera-view ${camera}`}>
          <video ref={videoRef} autoPlay muted playsInline onLoadedMetadata={e => { const v = e.currentTarget; if (v.parentElement && v.videoWidth && v.videoHeight) v.parentElement.style.aspectRatio = `${v.videoWidth} / ${v.videoHeight}` }} />
          {camera === 'ready' && <canvas ref={canvasRef} className="tracking-canvas" aria-hidden="true" />}
          {camera === 'ready' && stage !== 1 && <div className={`guide-outline stage-${stage}`}><span /></div>}
          {camera === 'ready' && stage === 1 && <div className="height-prompt"><label>Your height<span><input aria-label="Height in centimetres" type="number" min="100" max="250" value={height} onChange={e => setHeight(e.target.value)} placeholder="165" /><em>cm</em></span></label><small>Enter your actual height so the scan can be scaled accurately.</small></div>}
          {camera === 'idle' && <div className="camera-message"><span><Icon name="camera" /></span><h2>Camera access needed</h2><p>Your live preview stays in this browser.</p><Button onClick={startCamera}>Enable camera</Button></div>}
          {camera === 'loading' && <div className="camera-message"><span className="spinner" /><h2>Starting camera…</h2><p>Check your browser permission prompt.</p></div>}
          {(camera === 'denied' || camera === 'error') && <div className="camera-message error"><h2>{camera === 'denied' ? 'Camera permission blocked' : 'Camera unavailable'}</h2><p>{camera === 'denied' ? 'Allow camera access in browser settings, then retry.' : 'Close other camera apps or use manual entry.'}</p><Button onClick={startCamera}>Try again</Button></div>}
        </div>
        <div className="capture-bar">
          <Button disabled={camera !== 'ready' || (stage === 1 && !height)} onClick={nextCapture}>{currentStage.action} <Icon name={stage === 4 ? 'check' : 'arrow'} /></Button>
          <button className="text-button" onClick={() => { setStage(0); setHeight('') }}>Reset guide</button>
          <button className="text-button" onClick={stopCamera}>Stop camera</button>
        </div>
      </section>
      <aside className="tips-card">
        <div className="progress-head"><span>Scan progress</span><b>{(stage + 1) * 20}%</b></div>
        <div className="progress"><i style={{ width: `${(stage + 1) * 20}%` }} /></div>
        <h2>{currentStage.title}</h2>
        <p>{currentStage.detail}</p>
        <ol className="scan-checklist">
          {scanStages.map((item, index) => <li className={index === stage ? 'current' : index < stage ? 'complete' : ''} key={item.short}>
            <span>{index < stage ? <Icon name="check" /> : index + 1}</span><b>{item.short}</b>
          </li>)}
        </ol>
        <div className="detection-note"><Icon name="camera" /><span><b>Live body tracking</b><small role="status">{trackingMessage}</small><small>Guide progress is not measurement capture. Calibration and silhouette measurements are not connected yet.</small></span></div>
        <button className="manual-link" onClick={() => { stopCamera(); setManual(true) }}>Can’t use your camera? <b>Enter manually</b></button>
      </aside>
    </div>
  </main>
}

function Review({ go }: { go: (s: Step) => void }) {
  const [revealed, setRevealed] = useState(false)
  return <main className="page narrow-wide">
    <div className="page-title"><span className="eyebrow">Demo preview — no measurements captured</span><h1>Your body profile</h1><p>This screen previews the design. Real scan measurements are not connected yet.</p></div>
    <div className="review-grid">
      <section className="card silhouette-card"><Silhouette /><div className="confidence"><span><Icon name="check" /></span><div><small>Scan confidence</small><b>Not evaluated</b></div></div></section>
      <section className="card measurements">
        <div className="card-head"><div><span className="demo-pill">Demo mode</span><h2>Your measurements</h2></div><button className="reveal" onClick={() => setRevealed(!revealed)}><Icon name="eye" />{revealed ? 'Hide' : 'Reveal'}</button></div>
        <p>Measurements aren’t detected in this browser prototype. The values below are illustrative only.</p>
        {['Bust', 'Waist', 'Hips', 'Inseam'].map(label => <div className="measure-row" key={label}><span>{label}</span><b>{revealed ? 'Demo value' : '••••••••'}</b></div>)}
        <div className="review-actions"><Button onClick={() => go('import')}>Add a garment <Icon name="arrow" /></Button><Button secondary onClick={() => go('scan')}>Scan again</Button></div>
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

function Compare({ go }: { go: (s: Step) => void }) {
  const [size, setSize] = useState('20')
  return <main className="page compare-page">
    <div className="page-title split-title"><div><span className="eyebrow">Fit comparison · Demo mode</span><h1>How this dress may fit</h1></div><Button secondary onClick={() => go('import')}>Change garment</Button></div>
    <div className="size-tabs" role="tablist" aria-label="Garment size">{['16', '18', '20', '22', '24'].map(s => <button role="tab" aria-selected={size === s} onClick={() => setSize(s)} key={s}>Size {s}{s === '20' && <small>Best match</small>}</button>)}</div>
    <div className="compare-grid">
      <section className="card heatmap-card"><span className="demo-pill">Illustrative fit map</span><Silhouette heatmap /><div className="legend">{[['tight','Tight'],['snug','Snug'],['comfortable','Comfortable'],['loose','Loose']].map(([c,l]) => <span key={c}><i className={c}/>{l}</span>)}</div></section>
      <section className="recommendation">
        <div className="rec-top"><span><Icon name="spark" /></span><div><small>Sample size selection</small><h2>{size === '20' ? '20' : size}</h2></div><em>Demo mode</em></div>
        <h3>{size === '20' ? 'Balanced comfort with room to move' : `Viewing the sample fit for size ${size}`}</h3>
        <p>This illustrative recommendation uses sample data—not your scan. A real recommendation requires adapted in-browser pose estimation and a verified brand size chart.</p>
        <div className="fit-note"><b>Why this size?</b><span>Sample shaping allows more room through the waist while keeping the shoulders closer to the body.</span></div>
        <Button onClick={() => go('questions')}>Ask about this fit <Icon name="arrow" /></Button>
      </section>
    </div>
  </main>
}

function Questions({ go }: { go: (s: Step) => void }) {
  const [area, setArea] = useState('waist')
  const [question, setQuestion] = useState('')
  return <main className="page questions-page">
    <div className="page-title"><span className="eyebrow">Fit questions · Demo mode</span><h1>Where would you like more detail?</h1><p>Select an area on the body, then ask what matters to you.</p></div>
    <div className="question-grid">
      <section className="card question-body"><Silhouette heatmap onArea={setArea} /><p>Tap a marker to choose a body area</p></section>
      <section className="card ask-card">
        <span className="selected-area">Selected area <b>{area}</b></span>
        <h2>What would you like to know?</h2>
        <div className="suggestions">{[`Will this feel restrictive at my ${area}?`, `Will the fabric cling here?`, 'Could I sit comfortably?'].map(q => <button key={q} onClick={() => setQuestion(q)}>{q}<Icon name="chevron" /></button>)}</div>
        <label>Your question<textarea value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask about comfort, movement, or coverage…" /></label>
        <Button disabled={!question}>Get fit guidance <Icon name="arrow" /></Button>
        <p className="demo-note">Demo mode: answers require a verified garment model and browser-adapted scan data.</p>
        <button className="text-button" onClick={() => go('compare')}>Back to fit comparison</button>
      </section>
    </div>
  </main>
}

export default function App() {
  const [step, setStep] = useState<Step>('welcome')
  const go = (next: Step) => { setStep(next); window.scrollTo({ top: 0, behavior: 'smooth' }) }
  return <div className="app"><Header step={step} go={go} />{step === 'welcome' && <Welcome go={go} />}{step === 'scan' && <Scan go={go} />}{step === 'review' && <Review go={go} />}{step === 'import' && <ImportGarment go={go} />}{step === 'compare' && <Compare go={go} />}{step === 'questions' && <Questions go={go} />}<footer>Fit guidance, not a promise. Every body and garment is different.</footer></div>
}
