import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(here, '..')
const envFile = path.join(root, '.env.local')
const outFile = path.join(root, 'data', 'scrape.json')

function token() {
  const line = fs.readFileSync(envFile, 'utf8').split('\n').find(entry => entry.startsWith('VITE_APIFY_TOKEN='))
  return line?.slice('VITE_APIFY_TOKEN='.length).trim().replace(/^['"]|['"]$/g, '') || ''
}

const pageFunction = `async function pageFunction(context) {
  const { page, request } = context;
  await page.waitForTimeout(3000);
  const data = await page.evaluate(() => {
    const clean = (value) => (value || '').replace(/\\s+/g, ' ').trim();
    const tables = [...document.querySelectorAll('table')].slice(0, 20).map((table) =>
      [...table.querySelectorAll('tr')].slice(0, 40).map((row) =>
        [...row.querySelectorAll('th,td')].map((cell) => clean(cell.innerText)).filter(Boolean)
      ).filter((row) => row.length)
    ).filter((table) => table.length);
    const image = document.querySelector('meta[property="og:image"]')?.getAttribute('content')
      || document.querySelector('img')?.src
      || '';
    return {
      title: clean(document.querySelector('h1')?.innerText || document.title),
      image,
      text: clean(document.body.innerText).slice(0, 16000),
      tables,
    };
  });
  return { url: request.url, ...data };
}`

const input = {
  startUrls: [
    { url: 'https://www.levi.com/US/en_US/info/sizeguide' },
    { url: 'https://www.levi.com/US/en_US/clothing/women/outerwear/jean-jackets/original-trucker-jacket/p/299450031' },
    { url: 'https://www.levi.com/US/en_US/clothing/women/jeans/straight/501-original-fit-womens-jeans/p/125010438' },
    { url: 'https://therunhouse.com/products/nike-women-s-tech-fleece-full-zip-hoodie-nkfq4798' },
    { url: 'https://www.filson.com/products/womens-cotton-crewneck-sweater-copper-brown' },
  ],
  pageFunction,
  maxRequestsPerCrawl: 5,
  proxyConfiguration: { useApifyProxy: true },
}

const key = token()
if (!key) throw new Error('VITE_APIFY_TOKEN is missing')

const started = await fetch('https://api.apify.com/v2/acts/apify~playwright-scraper/runs', {
  method: 'POST',
  headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
  body: JSON.stringify(input),
})
const startedBody = await started.json()
if (!started.ok) throw new Error(JSON.stringify(startedBody.error || startedBody).slice(0, 400))
const runId = startedBody.data.id
const datasetId = startedBody.data.defaultDatasetId
console.log('RUN', runId, startedBody.data.status)

let status = startedBody.data.status
for (let i = 0; i < 40 && !['SUCCEEDED', 'FAILED', 'ABORTED', 'TIMED-OUT'].includes(status); i++) {
  await new Promise(resolve => setTimeout(resolve, 8000))
  const check = await fetch(`https://api.apify.com/v2/actor-runs/${runId}`, { headers: { Authorization: `Bearer ${key}` } })
  const checkBody = await check.json()
  status = checkBody.data.status
  console.log('STATUS', status)
}

const itemsResponse = await fetch(`https://api.apify.com/v2/datasets/${datasetId}/items`, { headers: { Authorization: `Bearer ${key}` } })
const items = await itemsResponse.json()
fs.mkdirSync(path.dirname(outFile), { recursive: true })
fs.writeFileSync(outFile, JSON.stringify(items, null, 2))
console.log('DATASET_READY', Array.isArray(items) ? items.length : 0, status)
