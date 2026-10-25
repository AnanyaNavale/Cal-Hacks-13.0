import { defineConfig, loadEnv, type HtmlTagDescriptor, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import fs from 'node:fs'
import path from 'node:path'

import siteConfiguration from './.figma/make/site.json'
import { attachLive } from './server/live'


// Vite config — https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // .figma/make/deploy-preview passes `--mode development` for cached-preview builds.
  const emitSourcemaps = mode === 'development'
  const env = loadEnv(mode, __dirname, '')

  return {
    base: process.env.FIGMA_PUBLIC_URL ? `${process.env.FIGMA_PUBLIC_URL}/` : '/',
    build: {
      sourcemap: emitSourcemaps ? 'inline' : false,
      minify: !emitSourcemaps,
    },
    plugins: [
      demoApi(env),
react(),
      tailwindcss(),
      figmaSiteConfiguration(siteConfiguration),
      figmaErrorOverlayReplay(),
      figmaReactRefreshBoundaryFallback(),
      figmaMakeKitPlugin({ storiesGlob: '/src/**/*.stories.{ts,tsx,js,jsx}' }),
      figmaLinkedPackageSourcemaps(),
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      host: process.env.FIGMA_DEV_SERVER_HOST || '0.0.0.0',
      port: parseInt(process.env.PORT || '8443'),
      strictPort: true,
      watch: {
        ignored: [
          '**/.figma/**',
],
      },
    },
    preview: {
      host: process.env.FIGMA_DEV_SERVER_HOST || '0.0.0.0',
      port: parseInt(process.env.PORT || '8443'),
    },
  }
})

function readBody(req: import('node:http').IncomingMessage) {
  return new Promise<string>((resolve, reject) => {
    const chunks: Buffer[] = []
    req.on('data', chunk => chunks.push(Buffer.from(chunk)))
    req.on('end', () => resolve(Buffer.concat(chunks).toString()))
    req.on('error', reject)
  })
}

function demoApi(env: Record<string, string>): Plugin {
  return {
    name: 'booth-demo-api',
    configureServer(server) {
      attachLive(server, __dirname)
      server.middlewares.use(async (req, res, next) => {
        const url = req.url?.split('?')[0]
        if (req.method !== 'POST' || (url !== '/api/ask' && url !== '/api/speak')) return next()
        try {
          const body = JSON.parse(await readBody(req) || '{}') as { text?: string; area?: string; question?: string; garment?: string; size?: string; covers?: string[]; wearer?: string }
          if (url === '/api/speak') {
            const key = env.CARTESIA_API_KEY
            if (!key) {
              res.statusCode = 503
              res.end('Add CARTESIA_API_KEY to figma-web/.env.local')
              return
            }
            const upstream = await fetch('https://api.cartesia.ai/tts/bytes', {
              method: 'POST',
              headers: {
                Authorization: `Bearer ${key}`,
                'Cartesia-Version': '2026-08-14',
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                model_id: 'sonic-3.6',
                transcript: String(body.text || '').slice(0, 500),
                voice: 'db6b0ed5-d5d3-463d-ae85-518a07d3c2b4',
                output_format: { container: 'wav', encoding: 'pcm_s16le', sample_rate: 44100 },
                language: 'en',
              }),
            })
            if (!upstream.ok) {
              res.statusCode = upstream.status
              res.end(await upstream.text())
              return
            }
            res.setHeader('Content-Type', 'audio/wav')
            res.end(Buffer.from(await upstream.arrayBuffer()))
            return
          }
          const key = env.ANTHROPIC_API_KEY
          if (!key) {
            res.statusCode = 503
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ error: 'Add ANTHROPIC_API_KEY to figma-web/.env.local, then restart the dev server.' }))
            return
          }
          const garment = String(body.garment || 'this garment').slice(0, 80)
          const size = String(body.size || '').slice(0, 40)
          const covers = Array.isArray(body.covers) ? body.covers.map(area => String(area)).join(', ') : String(body.area || '')
          const wearer = String(body.wearer || 'the measurements from this scan').slice(0, 300)
          const question = `Garment: ${garment}. Size: ${size}. Body area: ${body.area || 'general'}. Question: ${String(body.question || '').slice(0, 600)}`
          const payload = {
            model: 'claude-sonnet-5-5',
            max_tokens: 400,
            system: `You answer about one garment only: ${garment}, size ${size}. It touches these body areas: ${covers}. The wearer is ${wearer}. These measurements are webcam estimates, not a tape measure. Use only the measurements for the areas this garment covers. Do not mention any other garment, brand, or product. Answer in two or three short sentences. Do not give medical advice.`,
            messages: [{ role: 'user', content: question }],
          }
          let upstream = await fetch('https://api.anthropic.com/v1/messages', {
            method: 'POST',
            headers: { 'x-api-key': key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json' },
            body: JSON.stringify({ ...payload, output_config: { effort: 'low' } }),
          })
          if (upstream.status === 400) {
            upstream = await fetch('https://api.anthropic.com/v1/messages', {
              method: 'POST',
              headers: { 'x-api-key': key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json' },
              body: JSON.stringify(payload),
            })
          }
          const data = await upstream.json() as { content?: { type: string; text?: string }[]; error?: { message?: string } }
          res.statusCode = upstream.ok ? 200 : upstream.status
          res.setHeader('Content-Type', 'application/json')
          const answer = (data.content || []).filter(block => block.type === 'text').map(block => block.text || '').join('\n').trim()
          res.end(JSON.stringify(upstream.ok ? { answer: answer || 'No answer came back.' } : { error: data.error?.message || 'Anthropic could not answer.' }))
        } catch {
          res.statusCode = 500
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ error: 'The request failed.' }))
        }
      })
    },
  }
}

type FigmaSiteConfiguration = {
  title?: string
  description?: string
  language?: string
  robots?: {
    index?: boolean
  }
  icons?: {
    icon?: string
  }
  openGraph?: {
    image?: string
  }
  analytics?: {
    googleAnalyticsId?: string
  }
  customScripts?: {
    headStart?: string
    headEnd?: string
    bodyStart?: string
    bodyEnd?: string
  }
  accessibility?: {
    addBypassLinks?: boolean
  }
}

/** Applies /.figma/make/site.json to the generated document shell. */
function figmaSiteConfiguration(config: FigmaSiteConfiguration): Plugin {
  function sanitizeHtmlValue(value: string | undefined): string {
    return value?.replace(/[^a-zA-Z0-9_-]/g, '') || ''
  }
  function escapeHtmlText(value: string): string {
    return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  }
  function replaceHtmlCommentSlot(html: string, slotName: string, content: string): string {
    return html.replace(`<!-- ${slotName} -->`, content)
  }

  const title = config.title ?? "Figma Make App"
  const description = config.description ?? ''
  // Imported HTML keeps its own favicon; new Makes default to the Make product icon.
  const favicon = config.icons?.icon ?? '/favicon.svg'
  const socialImage = config.openGraph?.image ?? ''
  const language = sanitizeHtmlValue(config.language) || 'en'
  const googleAnalyticsId = sanitizeHtmlValue(config.analytics?.googleAnalyticsId)
  const headStart = config.customScripts?.headStart ?? ''
  const headEnd = config.customScripts?.headEnd ?? ''
  const bodyStart = config.customScripts?.bodyStart ?? ''
  const bodyEnd = config.customScripts?.bodyEnd ?? ''
  const robotsTxt = config.robots?.index === false ? 'User-agent: *\nDisallow: /\n' : ''

  return {
    name: 'figma-site-configuration',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!robotsTxt || req.url?.split('?')[0] !== '/robots.txt') return next()

        res.setHeader('Content-Type', 'text/plain; charset=utf-8')
        res.end(robotsTxt)
      })
    },
    generateBundle() {
      if (!robotsTxt) return

      this.emitFile({
        type: 'asset',
        fileName: 'robots.txt',
        source: robotsTxt,
      })
    },
    transformIndexHtml: {
      order: 'pre',
      handler(html) {
        let result = html
        result = replaceHtmlCommentSlot(result, 'figma:lang', language)
        result = replaceHtmlCommentSlot(result, 'figma:title', escapeHtmlText(title))
        result = replaceHtmlCommentSlot(result, 'figma:head-start', headStart)
        result = replaceHtmlCommentSlot(result, 'figma:head-end', headEnd)
        result = replaceHtmlCommentSlot(result, 'figma:body-start', bodyStart)
        result = replaceHtmlCommentSlot(result, 'figma:body-end', bodyEnd)

        const tags: HtmlTagDescriptor[] = []
        if (description) {
          tags.push({ tag: 'meta', attrs: { name: 'description', content: description }, injectTo: 'head' })
        }
        if (config.robots?.index === false) {
          tags.push({ tag: 'meta', attrs: { name: 'robots', content: 'noindex, nofollow' }, injectTo: 'head' })
        }
        if (favicon) {
          tags.push({ tag: 'link', attrs: { rel: 'icon', href: favicon }, injectTo: 'head' })
        }
        if (title) {
          tags.push({ tag: 'meta', attrs: { property: 'og:title', content: title }, injectTo: 'head' })
        }
        if (description) {
          tags.push({ tag: 'meta', attrs: { property: 'og:description', content: description }, injectTo: 'head' })
        }
        if (socialImage) {
          tags.push(
            { tag: 'meta', attrs: { property: 'og:image', content: socialImage }, injectTo: 'head' },
            { tag: 'meta', attrs: { name: 'twitter:card', content: 'summary_large_image' }, injectTo: 'head' },
            { tag: 'meta', attrs: { name: 'twitter:image', content: socialImage }, injectTo: 'head' },
          )
        }

        if (googleAnalyticsId) {
          tags.push(
            {
              tag: 'script',
              attrs: {
                async: true,
                src: `https://www.googletagmanager.com/gtag/js?id=${googleAnalyticsId}`,
              },
              injectTo: 'head',
            },
            {
              tag: 'script',
              children: `
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', ${JSON.stringify(googleAnalyticsId)});
`,
              injectTo: 'head',
            },
          )
        }

        if (config.accessibility?.addBypassLinks) {
          tags.push(
            {
              tag: 'style',
              children: `
  .figma-bypass-link {
    position: fixed;
    top: 8px;
    left: 8px;
    z-index: 2147483647;
    transform: translateY(-150%);
    border-radius: 6px;
    background: #111827;
    color: #fff;
    padding: 8px 12px;
    font: 600 14px/1.2 system-ui, sans-serif;
    text-decoration: none;
  }
  .figma-bypass-link:focus {
    transform: translateY(0);
  }
`,
              injectTo: 'head',
            },
            {
              tag: 'a',
              attrs: { class: 'figma-bypass-link', href: '#root' },
              children: 'Skip to content',
              injectTo: 'body-prepend',
            },
          )
        }

        return {
          html: result,
          tags,
        }
      },
    },
  }
}

/**
 * Replay the most recent build error to clients that connect after
 * it was first broadcast. Vite buffers an error payload only while
 * no clients are connected and clears the buffer on the first
 * reconnect (see `bufferedMessage` in `createWebSocketServer`), so
 * if the preview iframe reloads after Vite already delivered an
 * error to a live socket, the new socket misses the payload and
 * the overlay stays hidden even though the build is still broken.
 * We intercept `ws.send` to remember the latest error and replay
 * it on every new connection; the cache clears on a successful
 * `update` or `full-reload` so a stale overlay can't survive a
 * fixed build.
 */
function figmaErrorOverlayReplay(): Plugin {
  return {
    name: 'figma-error-overlay-replay',
    apply: 'serve',
    configureServer(server) {
      let lastError: object | null = null

      const origSend = server.ws.send.bind(server.ws) as (...args: any[]) => void
      server.ws.send = ((...args: any[]) => {
        const payload = args[0]
        if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
          const type = (payload as { type?: string }).type
          if (type === 'error') {
            lastError = payload as object
          } else if (type === 'update' || type === 'full-reload') {
            lastError = null
          }
        }
        return origSend(...args)
      }) as typeof server.ws.send

      server.ws.on('connection', (socket) => {
        if (lastError !== null) {
          socket.send(JSON.stringify(lastError))
        }
      })
    },
  }
}

/**
 * Reload when a module that previously defined a React Refresh boundary stops
 * defining one. This happens when an agent moves a component into a new file
 * and replaces the old module with a re-export:
 *
 *   export { default } from './app/App'
 *
 * Vite otherwise accepts the update using the previous module's HMR boundary,
 * but the re-export-only transform no longer registers a replacement for the
 * mounted component family. React reports a successful refresh while leaving
 * the old tree mounted until the page is reloaded.
 */
function figmaReactRefreshBoundaryFallback(): Plugin {
  type ModuleNode = import('vite').ModuleNode
  const hadRefreshBoundary = new Map<string, boolean>()
  const lostRefreshBoundaries = new Set<string>()
  let sendFullReload: (() => void) | null = null

  return {
    name: 'figma-react-refresh-boundary-fallback',
    apply: 'serve',
    enforce: 'post',
    configureServer(server) {
      sendFullReload = () => server.ws.send({ type: 'full-reload', path: '*' })
    },
    handleHotUpdate({ modules, server, timestamp }) {
      if (lostRefreshBoundaries.size === 0) return

      const visited = new Set<ModuleNode>()
      const pending = [...modules]
      while (pending.length > 0) {
        const current = pending.pop()
        if (!current || visited.has(current)) continue
        visited.add(current)

        const moduleId = current.id?.split('?')[0]
        if (moduleId && lostRefreshBoundaries.has(moduleId)) {
          const invalidated = new Set<ModuleNode>()
          for (const updatedModule of modules) {
            server.moduleGraph.invalidateModule(updatedModule, invalidated, timestamp, true)
          }
          sendFullReload?.()
          return []
        }

        pending.push(...current.importers)
      }
    },
    transform(code, id) {
      if (!/\.[jt]sx?(?:\?|$)/.test(id) || id.includes('/node_modules/')) return null

      const moduleId = id.split('?')[0] ?? id
      const hasRefreshBoundary = code.includes('registerExportsForReactRefresh')
      const previousHadRefreshBoundary = hadRefreshBoundary.get(moduleId)
      hadRefreshBoundary.set(moduleId, hasRefreshBoundary)

      if (hasRefreshBoundary) lostRefreshBoundaries.delete(moduleId)
      if (previousHadRefreshBoundary && !hasRefreshBoundary) {
        lostRefreshBoundaries.add(moduleId)
        queueMicrotask(() => sendFullReload?.())
      }

      return null
    },
  }
}

/**
 * Serves a blank render-target page at /.figma/make/kit.html that
 * the Figma preview script drives directly. The page exposes a
 * registry of every file matching `storiesGlob` on
 * window.__FIGMA__.stories so the design surface can dynamically
 * import + mount each entry into its own grid view.
 *
 * Dev-only: `apply: 'serve'` gates the plugin to `vite dev`. Prod
 * builds (`vite build`) skip it entirely so the route doesn't leak
 * into shipped bundles.
 */
function figmaMakeKitPlugin(options: { storiesGlob: string | string[] }): Plugin {
  const storiesGlob = Array.isArray(options.storiesGlob) ? options.storiesGlob : [options.storiesGlob]
  const ROUTE = '/.figma/make/kit.html'
  const VIRTUAL_ID = 'virtual:figma-stories'
  const RESOLVED_ID = '\0' + VIRTUAL_ID
  const STORIES_MODULE = `export const stories = import.meta.glob(${JSON.stringify(storiesGlob)})`
  const HTML_BOOTSTRAP = `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
</head>
<body>
<div id="figma-make-kit-root"></div>
<script type="module">
  import { stories } from 'virtual:figma-stories'
  window.__FIGMA__ = Object.assign(window.__FIGMA__ ?? {}, { stories })
  window.dispatchEvent(new CustomEvent('figma.ready'))
</script>
</body>
</html>`

  return {
    name: 'figma-make-kit',
    apply: 'serve',
    resolveId(id) {
      if (id === VIRTUAL_ID) return RESOLVED_ID
      return null
    },
    load(id) {
      if (id !== RESOLVED_ID) return null
      return STORIES_MODULE
    },
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = req.url || ''
        if (url.split('?')[0] !== ROUTE) return next()

        try {
          res.setHeader('Content-Type', 'text/html')
          res.end(await server.transformIndexHtml(url, HTML_BOOTSTRAP))
        } catch (err) {
          next(err as Error)
        }
      })
    },
  }
}

/**
 * Reports the sources of dependencies linked from outside node_modules (pnpm
 * or npm workspaces, `file:`, `link:`) under `node_modules/<name>/`, the way
 * installed packages appear in sourcemaps. Vite otherwise reports their real
 * path, such as `libraries/<name>/`.
 */
function figmaLinkedPackageSourcemaps(): Plugin {
  const packageJson = JSON.parse(fs.readFileSync(path.resolve(__dirname, 'package.json'), 'utf8'))
  const linkedPackages = Object.keys({ ...packageJson.dependencies, ...packageJson.devDependencies }).flatMap((name) => {
    const installedDir = path.resolve(__dirname, 'node_modules', name)
    try {
      const realDir = fs.realpathSync(installedDir)
      return realDir.split(path.sep).includes('node_modules') ? [] : [{ installedDir, realDir }]
    } catch {
      return []
    }
  })
  const sourceMappingUrl = /\/\/# sourceMappingURL=(\S+)\s*$/

  function installedPath(file: string) {
    const linked = linkedPackages.find(({ realDir }) => file.startsWith(realDir + path.sep))
    return linked && path.join(linked.installedDir, path.relative(linked.realDir, file))
  }
  function readSourceMap(file: string, code: string) {
    const url = sourceMappingUrl.exec(code)?.[1]
    try {
      if (url?.startsWith('data:')) {
        return { map: JSON.parse(Buffer.from(url.split(',')[1], 'base64').toString()), mapDir: path.dirname(file) }
      }
      if (url) {
        const mapFile = path.resolve(path.dirname(file), url)
        return { map: JSON.parse(fs.readFileSync(mapFile, 'utf8')), mapDir: path.dirname(mapFile) }
      }
    } catch {}
    // Without a usable input map, map every line to itself.
    const mappings = code.split('\n').map((_, i) => (i ? 'AACA' : 'AAAA'))
    const map = { version: 3, sources: [file], sourcesContent: [code], names: [], mappings: mappings.join(';') }
    return { map, mapDir: path.dirname(file) }
  }

  return {
    name: 'figma-linked-package-sourcemaps',
    enforce: 'pre',
    load(id) {
      // Leave queried imports such as ?raw and ?url to Vite's own loaders.
      if (id.includes('?') || !/\.[cm]?[jt]sx?$/.test(id) || !installedPath(id)) return null

      const code = fs.readFileSync(id, 'utf8')
      const { map, mapDir } = readSourceMap(id, code)
      const sourceRoot = path.resolve(mapDir, map.sourceRoot ?? '')
      map.sources = map.sources.map((source: string) => {
        const original = path.resolve(sourceRoot, source)
        return installedPath(original) ?? original
      })
      delete map.sourceRoot
      return { code: code.replace(sourceMappingUrl, ''), map }
    },
  }
}
