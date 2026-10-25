import { spawn, type ChildProcess } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import type { IncomingMessage, ServerResponse } from 'node:http'
import type { ViteDevServer } from 'vite'

let webRoot = path.resolve(process.cwd())
let repoRoot = path.resolve(webRoot, '..')
let dbPath = path.join(webRoot, 'data', 'catalog.sqlite')
const python = path.join(repoRoot, '.venv', 'bin', 'python')
const bridge = path.join(repoRoot, 'scanner_bridge.py')

export type SizeRow = {
  label: string
  bustMin: number | null
  bustMax: number | null
  waistMin: number | null
  waistMax: number | null
  hipsMin: number | null
  hipsMax: number | null
}

export type CatalogProduct = {
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

type ScanCue = { step: string; title: string; say: string; hint: string; progress: number | null }
type ScanJob = { status: 'idle' | 'running' | 'complete' | 'error' | 'cancelled'; profile?: unknown; error?: string; scan?: ScanCue | null }

let child: ChildProcess | null = null
let job: ScanJob = { status: 'idle' }
let statusPath: string | null = null
let framePath: string | null = null

function database() {
  fs.mkdirSync(path.dirname(dbPath), { recursive: true })
  const db = new DatabaseSync(dbPath)
  db.exec(`
    CREATE TABLE IF NOT EXISTS products (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      brand TEXT NOT NULL,
      source_url TEXT NOT NULL,
      image_path TEXT,
      price TEXT,
      areas TEXT NOT NULL,
      fit_note TEXT,
      inseam_in REAL,
      scraped_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS size_rows (
      product_id TEXT NOT NULL,
      label TEXT NOT NULL,
      bust_min REAL,
      bust_max REAL,
      waist_min REAL,
      waist_max REAL,
      hips_min REAL,
      hips_max REAL,
      sort_order INTEGER NOT NULL,
      PRIMARY KEY (product_id, label)
    );
  `)
  return db
}

export function readCatalog(): CatalogProduct[] {
  const db = database()
  const products = db.prepare('SELECT * FROM products ORDER BY rowid').all() as Array<Record<string, string | number | null>>
  const rows = db.prepare('SELECT * FROM size_rows ORDER BY sort_order').all() as Array<Record<string, string | number | null>>
  db.close()
  return products.map(product => ({
    id: String(product.id),
    name: String(product.name),
    brand: String(product.brand),
    url: String(product.source_url),
    image: String(product.image_path || ''),
    price: String(product.price || ''),
    areas: JSON.parse(String(product.areas || '[]')) as string[],
    fitNote: String(product.fit_note || ''),
    inseamIn: product.inseam_in == null ? null : Number(product.inseam_in),
    rows: rows.filter(row => row.product_id === product.id).map(row => ({
      label: String(row.label),
      bustMin: num(row.bust_min),
      bustMax: num(row.bust_max),
      waistMin: num(row.waist_min),
      waistMax: num(row.waist_max),
      hipsMin: num(row.hips_min),
      hipsMax: num(row.hips_max),
    })),
  }))
}

function num(value: string | number | null) {
  return value == null || value === '' ? null : Number(value)
}

export function replaceCatalog(products: Array<CatalogProduct & { scrapedAt: string }>) {
  const db = database()
  db.exec('DELETE FROM size_rows; DELETE FROM products;')
  const insertProduct = db.prepare(`INSERT INTO products
    (id, name, brand, source_url, image_path, price, areas, fit_note, inseam_in, scraped_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
  const insertRow = db.prepare(`INSERT INTO size_rows
    (product_id, label, bust_min, bust_max, waist_min, waist_max, hips_min, hips_max, sort_order)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`)
  for (const product of products) {
    insertProduct.run(product.id, product.name, product.brand, product.url, product.image, product.price, JSON.stringify(product.areas), product.fitNote, product.inseamIn, product.scrapedAt)
    product.rows.forEach((row, index) => {
      insertRow.run(product.id, row.label, row.bustMin, row.bustMax, row.waistMin, row.waistMax, row.hipsMin, row.hipsMax, index)
    })
  }
  db.close()
}

function send(res: ServerResponse, status: number, body: unknown) {
  const encoded = JSON.stringify(body)
  res.statusCode = status
  res.setHeader('Content-Type', 'application/json')
  res.end(encoded)
}

async function readJson(req: IncomingMessage) {
  const chunks: Buffer[] = []
  for await (const chunk of req) chunks.push(Buffer.from(chunk))
  return JSON.parse(Buffer.concat(chunks).toString() || '{}') as { heightCm?: number }
}

function readCue(): ScanCue | null {
  if (!statusPath) return null
  try {
    return JSON.parse(fs.readFileSync(statusPath, 'utf8')) as ScanCue
  } catch {
    return null
  }
}

function clearScanFiles(output: string, statusFile: string, frameFile: string) {
  for (const file of [output, statusFile, frameFile, `${statusFile}.tmp`, `${frameFile}.tmp`]) {
    fs.rmSync(file, { force: true })
  }
  if (statusPath === statusFile) statusPath = null
  if (framePath === frameFile) framePath = null
}

function startScan(heightCm: number) {
  if (!fs.existsSync(python) || !fs.existsSync(bridge)) {
    job = { status: 'error', error: 'The Python scanner is not installed in .venv.' }
    return
  }
  const id = Date.now()
  const output = path.join(os.tmpdir(), `fit-scan-${id}.json`)
  const statusFile = path.join(os.tmpdir(), `fit-scan-${id}-status.json`)
  const frameFile = path.join(os.tmpdir(), `fit-scan-${id}-frame.jpg`)
  statusPath = statusFile
  framePath = frameFile
  job = { status: 'running', scan: null }
  const process = spawn(python, [bridge, output, String(heightCm), statusFile, frameFile], { cwd: repoRoot })
  child = process
  let stderr = ''
  process.stderr.on('data', chunk => { stderr = (stderr + chunk.toString()).slice(-2000) })
  const timer = setTimeout(() => {
    process.kill()
    if (job.status === 'running') job = { status: 'error', error: 'The scan timed out. Close the scanner window and try again.' }
  }, 8 * 60 * 1000)
  process.on('close', code => {
    clearTimeout(timer)
    child = null
    if (job.status === 'cancelled') {
      clearScanFiles(output, statusFile, frameFile)
      return
    }
    try {
      if (!fs.existsSync(output)) throw new Error(stderr.trim() || 'The scanner closed before both steps finished.')
      const profile = JSON.parse(fs.readFileSync(output, 'utf8'))
      job = { status: 'complete', profile, scan: readCue() }
    } catch (error) {
      job = { status: 'error', error: error instanceof Error ? error.message : `Scanner exited with code ${code}.` }
    } finally {
      clearScanFiles(output, statusFile, frameFile)
    }
  })
}

export function attachLive(server: ViteDevServer, siteRoot: string) {
  webRoot = siteRoot
  repoRoot = path.resolve(siteRoot, '..')
  dbPath = path.join(siteRoot, 'data', 'catalog.sqlite')
  server.middlewares.use(async (req, res, next) => {
    const url = req.url?.split('?')[0]
    if (url === '/api/catalog' && req.method === 'GET') {
      send(res, 200, { products: readCatalog() })
      return
    }
    if (url === '/api/scan' && req.method === 'GET') {
      send(res, 200, { ...job, scan: job.status === 'running' ? readCue() : job.scan ?? null })
      return
    }
    if (url === '/api/scan/frame' && req.method === 'GET') {
      if (!framePath || !fs.existsSync(framePath)) {
        res.statusCode = 204
        res.end()
        return
      }
      res.statusCode = 200
      res.setHeader('Content-Type', 'image/jpeg')
      res.setHeader('Cache-Control', 'no-store')
      res.end(fs.readFileSync(framePath))
      return
    }
    if (url === '/api/scan/cancel' && req.method === 'POST') {
      child?.kill()
      job = { status: 'cancelled' }
      send(res, 200, job)
      return
    }
    if (url === '/api/scan' && req.method === 'POST') {
      if (job.status === 'running') {
        send(res, 409, { error: 'A scan is already running.' })
        return
      }
      try {
        const body = await readJson(req)
        const heightCm = Number(body.heightCm)
        if (!(heightCm >= 100 && heightCm <= 250)) {
          send(res, 400, { error: 'Height must be between 100 and 250 cm.' })
          return
        }
        startScan(heightCm)
        send(res, job.status === 'error' ? 503 : 202, job)
      } catch {
        send(res, 400, { error: 'Could not start the scanner.' })
      }
      return
    }
    next()
  })
}
