"""Serve Fit and connect its UI to the local Python webcam scanner.
Run from the repository root: python run_fit.py
"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import importlib.util
import json
import secrets
import subprocess
import sys
import tempfile
import threading
import time

ROOT=Path(__file__).resolve().parent
DIST=ROOT/'figma-web'/'dist'
LOCK=threading.Lock()
TOKEN=secrets.token_urlsafe(24)
JOB={'status':'idle'}
PROCESS=None

def dependencies_available():
    return all(importlib.util.find_spec(name) is not None for name in ('cv2','mediapipe','numpy'))

def scan_worker(height):
    global JOB,PROCESS
    with tempfile.TemporaryDirectory(prefix='fit-scan-') as directory:
        output=Path(directory)/'profile.json'
        log=Path(directory)/'scanner.log'
        try:
            with log.open('w',encoding='utf-8') as stream:
                flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
                with LOCK:
                    if JOB.get('status')!='running': return
                    PROCESS=subprocess.Popen([sys.executable,str(ROOT/'backend'/'scanner_runner.py'),str(output),str(height)],cwd=ROOT,stdout=stream,stderr=stream,creationflags=flags)
                    process=PROCESS
                try:
                    code=process.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
                    raise RuntimeError('Scan timed out after five minutes. Restart and follow the on-screen instructions.')
            with LOCK:
                if JOB.get('status')=='cancelled': return
                if code != 0 or not output.exists():
                    raise RuntimeError('Python scan did not complete. Check camera access, close other camera apps and retry. The Python window must finish both capture steps.')
                profile=json.loads(output.read_text(encoding='utf-8'))
                JOB={'status':'complete','profile':profile}
        except Exception as exc:
            with LOCK:
                if JOB.get('status')!='cancelled': JOB={'status':'error','error':str(exc)}
        finally:
            with LOCK: PROCESS=None

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=str(DIST),**kwargs)
    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        super().end_headers()
    def json_response(self,data,status=200):
        encoded=json.dumps(data,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(encoded)));self.end_headers();self.wfile.write(encoded)
    def trusted_host(self):
        return self.headers.get('Host','').lower() in {f'localhost:{self.server.server_port}',f'127.0.0.1:{self.server.server_port}'}
    def do_GET(self):
        if not self.trusted_host(): self.json_response({'error':'Use the localhost address printed in the terminal.'},403);return
        path=urlsplit(self.path).path
        if path=='/api/health':
            self.json_response({'ok':True,'pythonScannerAvailable':dependencies_available(),'sessionToken':TOKEN});return
        if path=='/api/scan':
            if not secrets.compare_digest(self.headers.get('X-Fit-Session',''),TOKEN): self.json_response({'error':'Invalid session.'},403);return
            with LOCK: snapshot=dict(JOB)
            self.json_response(snapshot);return
        if path.startswith('/api/'):
            self.json_response({'error':'Not found.'},404);return
        if path=='/': self.path='/index.html'
        super().do_GET()
    def do_POST(self):
        global JOB,PROCESS
        origin=self.headers.get('Origin')
        allowed={f'http://localhost:{self.server.server_port}',f'http://127.0.0.1:{self.server.server_port}'}
        if not self.trusted_host() or (origin and origin not in allowed) or not secrets.compare_digest(self.headers.get('X-Fit-Session',''),TOKEN):
            self.json_response({'error':'Open Fit on this computer using its localhost address.'},403);return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length<0 or length>2048: raise ValueError('Request too large.')
            data=json.loads(self.rfile.read(length) or b'{}')
            if not isinstance(data,dict): raise ValueError('Expected an object.')
        except (ValueError,TypeError): self.json_response({'error':'Invalid request.'},400);return
        path=urlsplit(self.path).path
        if path=='/api/scan/cancel':
            with LOCK:
                if PROCESS is not None: PROCESS.terminate()
                JOB={'status':'cancelled'}
            self.json_response({'status':'cancelled'});return
        if path!='/api/scan': self.json_response({'error':'Not found.'},404);return
        try: height=float(data.get('heightCm',0))
        except (ValueError,TypeError): height=0
        if not 100<=height<=250: self.json_response({'error':'Height must be 100–250 cm.'},400);return
        if not dependencies_available(): self.json_response({'error':'Install Python scanner dependencies: python -m pip install -r backend/requirements.txt'},503);return
        with LOCK:
            if PROCESS is not None or JOB.get('status')=='running': self.json_response({'error':'A scan is already running.'},409);return
            JOB={'status':'running'}
        threading.Thread(target=scan_worker,args=(height,),daemon=True).start()
        self.json_response({'status':'running'},202)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    if not (DIST/'index.html').exists(): raise SystemExit('Build the website first: cd figma-web, npm install, npm run build')
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Fit is ready: http://localhost:{args.port}',flush=True)
    print('Keep this terminal open. Press Ctrl+C to stop.',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        with LOCK:
            if PROCESS is not None: PROCESS.terminate()

if __name__=='__main__': main()
