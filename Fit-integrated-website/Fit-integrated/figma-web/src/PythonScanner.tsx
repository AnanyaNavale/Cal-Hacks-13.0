import { useEffect, useRef, useState } from 'react';
import type { BodyProfile } from './bodyMeasurements';

export function PythonScanner({heightCm, beforeStart, onProfile}:{heightCm:number;beforeStart:()=>void;onProfile:(profile:BodyProfile)=>void}) {
  const [running,setRunning]=useState(false);
  const [height,setHeight]=useState('');
  const scanHeight=height ? Number(height) : heightCm;
  const [message,setMessage]=useState('Use your original Python scanner on this computer instead.');
  const timer=useRef<ReturnType<typeof setTimeout>|null>(null);
  const token=useRef('');
  const request=useRef(0);
  const completeRef=useRef(onProfile); completeRef.current=onProfile;
  useEffect(()=>()=>{request.current++;if(timer.current)clearTimeout(timer.current);if(token.current)void fetch('/api/scan/cancel',{method:'POST',headers:{'X-Fit-Session':token.current}}).catch(()=>{});},[]);
  async function cancel(){
    request.current++;
    if(timer.current)clearTimeout(timer.current);
    if(token.current)await fetch('/api/scan/cancel',{method:'POST',headers:{'X-Fit-Session':token.current}}).catch(()=>{});
    setRunning(false);setMessage('Python scan cancelled.');
  }
  async function start(){
    if(!(scanHeight>=100 && scanHeight<=250)){setMessage('Enter your height above (100–250 cm) first.');return;}
    const current=++request.current;
    setRunning(true);setMessage('Connecting to the local Python scanner...');
    try{
      const response=await fetch('/api/health');
      if(!response.ok || !response.headers.get('content-type')?.includes('application/json'))throw new Error('Start the website with python run_fit.py to use this option. Browser scanning works independently.');
      const health=await response.json();
      if(current!==request.current)return;
      token.current=health.sessionToken;
      if(!health.pythonScannerAvailable)throw new Error('Install the Python scanner dependencies using backend/requirements.txt, then retry.');
      beforeStart();
      const result=await fetch('/api/scan',{method:'POST',headers:{'Content-Type':'application/json','X-Fit-Session':token.current},body:JSON.stringify({heightCm:scanHeight})});
      const data=await result.json();
      if(!result.ok)throw new Error(data.error||'Could not start Python.');
      if(current!==request.current){void fetch('/api/scan/cancel',{method:'POST',headers:{'X-Fit-Session':token.current}});return;}
      setMessage('A Python camera window will open. Follow its A-pose and full-turn instructions; return here when it finishes.');
      async function poll(){
        if(current!==request.current)return;
        try{
          const response=await fetch('/api/scan',{headers:{'X-Fit-Session':token.current}});
          const state=await response.json();
          if(current!==request.current)return;
          if(!response.ok)throw new Error(state.error||'Could not read scanner status.');
          if(state.status==='complete') {setRunning(false);completeRef.current(state.profile);return;}
          if(state.status==='error' || state.status==='cancelled')throw new Error(state.error||'Scan cancelled.');
          timer.current=setTimeout(poll,1000);
        }catch(error){if(current===request.current){setRunning(false);setMessage(error instanceof Error?error.message:'Scanner connection lost.');}}
      }
      void poll();
    }catch(error){if(current===request.current){setRunning(false);setMessage(error instanceof Error?error.message:'Could not start Python scanner.');}}
  }
  return <div className="python-option"><b>Original Python scanner</b><label>Height for Python scan (cm)<input type="number" min="100" max="250" value={height || (heightCm ? String(heightCm) : '')} onChange={e=>setHeight(e.target.value)} disabled={running} /></label><p role="status">{message}</p><button className="btn btn-secondary" disabled={running} onClick={start}>Use Python scanner</button>{running && <button className="text-button" onClick={cancel}>Cancel Python scan</button>}</div>;
}
