import { useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import type { PoseLandmarker } from '@mediapipe/tasks-vision';
import { combine, framing, makeProfile, measureFront, measureSide, personMask, pitchCorrection, pixelPoints } from './bodyMeasurements';
import type { BodyProfile, Sample } from './bodyMeasurements';

type View = 'front' | 'side';
type Capture = { view: View; samples: Sample[]; holdAt: number; collectAt: number; lastGood: number; anchor: number[] | null };
export function useBodyScanner(enabled: boolean, videoRef: RefObject<HTMLVideoElement | null>, canvasRef: RefObject<HTMLCanvasElement | null>, heightCm: number) {
  const [message, setMessage] = useState('Enable your camera to begin.');
  const [ready, setReady] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [frontDone, setFrontDone] = useState(false);
  const [profile, setProfile] = useState<BodyProfile | null>(null);
  const heightRef=useRef(heightCm); heightRef.current=heightCm;
  const frontRef=useRef<Sample | null>(null);
  const captureRef=useRef<Capture | null>(null);
  const mounted=useRef(true);
  useEffect(()=>{ mounted.current=true; return ()=>{mounted.current=false;}; },[]);

  function reset() {
    captureRef.current=null; frontRef.current=null;
    setCapturing(false); setFrontDone(false); setProgress(0); setProfile(null);
  }
  function capture(view: View) {
    if (!ready) return;
    if (!Number.isFinite(heightRef.current) || heightRef.current<100 || heightRef.current>250) { setMessage('Enter your height between 100 and 250 cm.'); return; }
    if (view==='side' && !frontRef.current) { setMessage('Capture the front view first.'); return; }
    if (view==='front') { frontRef.current=null; setFrontDone(false); setProfile(null); }
    captureRef.current={view,samples:[],holdAt:0,collectAt:0,lastGood:0,anchor:null};
    setCapturing(true); setProgress(0); setMessage('Move into position, then hold still.');
  }

  useEffect(()=>{
    if (!enabled) { setReady(false); captureRef.current=null; setCapturing(false); return; }
    let cancelled=false, frame=0, model: PoseLandmarker | null=null;
    let lastVideo=-1, lastRun=-1;
    const reportFailure=(text:string, now:number)=>{
      setMessage(text);
      const c=captureRef.current;
      if (c) {
        c.holdAt=0;
        if (!c.lastGood || now-c.lastGood>500) { c.samples=[]; c.collectAt=0; c.anchor=null; setProgress(0); }
      }
    };
    async function initialize() {
      try {
        setReady(false); setMessage('Loading MediaPipe body scanner...');
        const {FilesetResolver,PoseLandmarker,DrawingUtils}=await import('@mediapipe/tasks-vision');
        const files=await FilesetResolver.forVisionTasks(`${import.meta.env.BASE_URL}wasm`);
        if(cancelled) return;
        const created=await PoseLandmarker.createFromOptions(files,{
          baseOptions:{modelAssetPath:`${import.meta.env.BASE_URL}models/pose_landmarker.task`},
          runningMode:'VIDEO',numPoses:1,outputSegmentationMasks:true,
          minPoseDetectionConfidence:0.6,minPosePresenceConfidence:0.6,minTrackingConfidence:0.6,
        });
        if(cancelled){created.close();return;}
        model=created; setReady(true); setMessage('Scanner ready. Enter height, then follow the guide.');
        function detect() {
          if(cancelled) return;
          try {
            const video=videoRef.current,canvas=canvasRef.current,ctx=canvas?.getContext('2d'),now=performance.now();
            if(video && canvas && ctx && video.readyState>=2 && video.currentTime!==lastVideo && now-lastRun>=100){
              lastVideo=video.currentTime;lastRun=now;
              if(canvas.width!==video.videoWidth || canvas.height!==video.videoHeight){canvas.width=video.videoWidth;canvas.height=video.videoHeight;}
              ctx.clearRect(0,0,canvas.width,canvas.height);
              created.detectForVideo(video,now,result=>{
                const points=result.landmarks[0];
                if(!points){reportFailure('No body detected. Adjust lighting and framing.',now);return;}
                const drawing=new DrawingUtils(ctx);
                drawing.drawConnectors(points,PoseLandmarker.POSE_CONNECTIONS,{color:'#2dd4bf',lineWidth:3});
                drawing.drawLandmarks(points,{color:'white',fillColor:'#0f766e',radius:3});
                const c=captureRef.current;
                if(!c){setMessage(frontRef.current?'Front captured. Turn sideways for the side scan.':'Body detected. Follow the guide and capture your front view.');return;}
                const raw=result.segmentationMasks?.[0];
                if(!raw){reportFailure('Person silhouette is unavailable. Restart the camera.',now);return;}
                const mask=personMask(raw.getAsFloat32Array(),raw.width,raw.height,video.videoWidth,video.videoHeight);
                if(!mask){reportFailure('Unable to isolate your silhouette. Try a plain background.',now);return;}
                const issue=framing(points,mask,c.view);
                if(issue){reportFailure(issue,now);return;}
                const p=pixelPoints(points,mask), vc=pitchCorrection(result.worldLandmarks[0]??[]);
                const sample=c.view==='front'?measureFront(p,mask,vc):measureSide(p,mask,frontRef.current!,vc);
                if(!sample){reportFailure('Silhouette unclear. Separate your arms from your torso and improve lighting.',now);return;}
                const anchor=[(points[23].x+points[24].x)/2,(points[23].y+points[24].y)/2,(mask.bottom-mask.top)/mask.height];
                const moved=c.anchor && anchor.some((v,i)=>Math.abs(v-c.anchor![i])>0.025);
                if(moved){c.samples=[];c.holdAt=0;c.collectAt=0;c.anchor=null;setProgress(0);}
                c.anchor=c.anchor??anchor;
                // Reject outlier widths/depths instead of smoothing bad measurements into results.
                if(c.samples.length>=5){
                  const center=combine(c.samples);
                  const keys=c.view==='front'?['chest','waist','hip','stature']:['bust','waist','hip'];
                  if(keys.some(k=>sample[k] && center[k] && Math.abs(sample[k]!/center[k]!-1)>0.08)){
                    reportFailure('Measurements are unstable. Hold still and check lighting.',now);return;
                  }
                }
                c.lastGood=now;
                if(!c.holdAt)c.holdAt=now;
                if(now-c.holdAt<2000){setMessage('Position looks good. Hold still for two seconds...');return;}
                if(!c.collectAt)c.collectAt=now;
                c.samples.push(sample);
                const target=c.view==='front'?45:30;
                setProgress(Math.min(99,Math.round(100*Math.min(c.samples.length/target,(now-c.collectAt)/3000))));
                setMessage(`Capturing ${c.view} silhouette. Keep still...`);
                if(c.samples.length>=target && now-c.collectAt>=3000){
                  const final=combine(c.samples);
                  captureRef.current=null;setCapturing(false);setProgress(100);
                  if(c.view==='front'){frontRef.current=final;setFrontDone(true);setMessage('Front captured. Continue to the side view.');}
                  else {setProfile(makeProfile(frontRef.current!,final,heightRef.current));setMessage('Front and side captured. Estimates are ready for review.');}
                }
              });
            }
            frame=requestAnimationFrame(detect);
          }catch(error){
            captureRef.current=null;setCapturing(false);setReady(false);
            model?.close();model=null;
            setMessage(`Scan stopped: ${error instanceof Error?error.message:'Unknown error'}. Stop and restart the camera.`);
          }
        }
        frame=requestAnimationFrame(detect);
      }catch(error){if(!cancelled)setMessage(`Scanner could not load: ${error instanceof Error?error.message:'Unknown error'}. Check your connection, then restart the camera.`);}
    }
    void initialize();
    return ()=>{cancelled=true;cancelAnimationFrame(frame);model?.close();captureRef.current=null;if(mounted.current)setCapturing(false);};
  },[enabled,videoRef,canvasRef]);
  return {message,ready,capturing,progress,frontDone,profile,capture,reset};
}
