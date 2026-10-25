export type Point = { x: number; y: number; z?: number; visibility?: number };
export type Zone = 'bust' | 'waist' | 'hip' | 'thigh' | 'shoulder' | 'inseam';
export interface BodyProfile {
  heightCm: number;
  measurementsCm: Partial<Record<Zone, number>>;
  confidence: Partial<Record<Zone, 'low'>>;
  scannedAt: string;
  source: 'webcam' | 'manual' | 'python';
  notes: string[];
}
export type Sample = Record<string, number | null>;
export type Mask = { data: Uint8Array; width: number; height: number; top: number; bottom: number };
const mid = (a: Point, b: Point): Point => ({ x: (a.x+b.x)/2, y: (a.y+b.y)/2 });
const distance = (a: Point, b: Point) => Math.hypot(a.x-b.x, a.y-b.y);
export function median(values: number[]): number {
  const a = [...values].sort((a,b) => a-b);
  const i = Math.floor(a.length/2);
  return a.length % 2 ? a[i] : (a[i-1]+a[i])/2;
}
export function combine(samples: Sample[]): Sample {
  const out: Sample = {};
  for (const key of Object.keys(samples[0])) {
    const values = samples.map(s => s[key]).filter((v): v is number => v !== null && Number.isFinite(v));
    out[key] = values.length ? median(values) : null;
  }
  return out;
}
// Resample the model mask into camera aspect ratio before measuring widths.
export function personMask(raw: Float32Array, mw: number, mh: number, vw: number, vh: number): Mask | null {
  const height = Math.min(480, vh);
  const width = Math.round(height * vw / vh);
  const binary = new Uint8Array(width*height);
  for (let y=0; y<height; y++) for (let x=0; x<width; x++) {
    binary[y*width+x] = raw[Math.min(mh-1, Math.floor(y*mh/height))*mw + Math.min(mw-1, Math.floor(x*mw/width))] > 0.5 ? 1 : 0;
  }
  const queue = new Int32Array(binary.length);
  let best: Int32Array | null = null;
  for (let seed=0; seed<binary.length; seed++) {
    if (!binary[seed]) continue;
    let head=0, tail=1;
    queue[0]=seed; binary[seed]=0;
    while (head<tail) {
      const i=queue[head++], x=i%width;
      const neighbors = [x>0 ? i-1 : -1, x<width-1 ? i+1 : -1, i>=width ? i-width : -1, i<binary.length-width ? i+width : -1];
      for (const n of neighbors) if (n>=0 && binary[n]) { binary[n]=0; queue[tail++]=n; }
    }
    if (!best || tail>best.length) best=queue.slice(0,tail);
  }
  if (!best || best.length < width*height*0.015) return null;
  const data = new Uint8Array(binary.length);
  let top=height, bottom=0;
  for (const i of best) { data[i]=1; const y=Math.floor(i/width); top=Math.min(top,y); bottom=Math.max(bottom,y); }
  return { data, width, height, top, bottom };
}
const inside = (m: Mask, x: number, y: number) => {
  x=Math.floor(x); y=Math.floor(y);
  return x>=0 && x<m.width && y>=0 && y<m.height && m.data[y*m.width+x]===1;
};
function rowWidth(m: Mask, y: number, x: number): number | null {
  y=Math.floor(y); x=Math.floor(x);
  if (!inside(m,x,y)) return null;
  let l=x, r=x;
  while (inside(m,l-1,y)) l--;
  while (inside(m,r+1,y)) r++;
  return r-l+1;
}
function band(m: Mask, y: number, x: number): number | null {
  const values=[-2,-1,0,1,2].map(d=>rowWidth(m,y+d,x)).filter((v): v is number=>v!==null);
  return values.length ? median(values) : null;
}
function perpendicular(m: Mask, a: Point, b: Point, t: number): number | null {
  const length=distance(a,b);
  if (length<1) return null;
  const nx=-(b.y-a.y)/length, ny=(b.x-a.x)/length;
  const cx=a.x+t*(b.x-a.x), cy=a.y+t*(b.y-a.y);
  if (!inside(m,cx,cy)) return null;
  function walk(sign: number) {
    for (let s=1;s<Math.max(m.width,m.height);s++) if (!inside(m,cx+sign*s*nx,cy+sign*s*ny)) return s;
    return Math.max(m.width,m.height);
  }
  return walk(1)+walk(-1)-1;
}
function torso(p: Point[]) {
  const shoulder=mid(p[11],p[12]), hip=mid(p[23],p[24]);
  const length=hip.y-shoulder.y;
  return { shoulder, hip, length, y: (f:number)=>shoulder.y+f*length, x: (f:number)=>shoulder.x+Math.min(f,1)*(hip.x-shoulder.x) };
}
export function pixelPoints(points: Point[], m: Mask): Point[] { return points.map(p=>({...p,x:p.x*m.width,y:p.y*m.height})); }
export function pitchCorrection(world: Point[]): number {
  if (world.length<29) return 1;
  const sy=(world[11].y+world[12].y)/2, sz=((world[11].z??0)+(world[12].z??0))/2;
  const ay=(world[27].y+world[28].y)/2, az=((world[27].z??0)+(world[28].z??0))/2;
  const pitch=Math.min(Math.atan2(Math.abs(sz-az),Math.abs(sy-ay)),35*Math.PI/180);
  return 1/Math.cos(pitch);
}
export function framing(points: Point[], m: Mask, view: 'front'|'side'): string | null {
  if (m.top<3 || m.bottom>=m.height-3) return 'Bring your entire head and feet inside the frame.';
  if (m.bottom-m.top<m.height*0.55) return 'Come slightly closer while keeping head and feet visible.';
  const required=view==='front' ? [0,11,12,13,14,15,16,23,24,25,26,27,28] : [0];
  if (required.some(i=>(points[i].visibility??0)<0.6 || points[i].x<0.02 || points[i].x>0.98 || points[i].y<0.02 || points[i].y>0.98)) return 'Keep your head, hands and feet clearly visible.';
  if (view==='side' && (Math.max(points[23].visibility??0,points[24].visibility??0)<0.5 || Math.max(points[27].visibility??0,points[28].visibility??0)<0.5)) return 'Keep your hip and feet visible in the side view.';
  const p=pixelPoints(points,m), t=torso(p), sw=distance(p[11],p[12]);
  if (t.length<30) return 'Stand upright and move into view.';
  if (view==='side') return sw/distance(t.shoulder,t.hip)>0.35 ? 'Turn 90 degrees to your side, with arms down.' : null;
  if (sw<20) return 'Face the camera squarely.';
  for (const [w,h] of [[15,23],[16,24]]) {
    if (Math.abs(p[w].x-p[h].x)<0.6*sw) return 'Move your arms away from your torso, diagonally DOWN in an A-pose.';
    if (p[w].y<t.shoulder.y+0.3*t.length) return 'Lower your arms to about 45 degrees below your shoulders.';
  }
  if (Math.abs(p[27].x-p[28].x)<0.3*sw) return 'Separate your feet to about shoulder width.';
  return null;
}
// Port of test.py front sampling: silhouette widths normalized to shoulder width.
export function measureFront(p: Point[], m: Mask, vc: number): Sample | null {
  const t=torso(p), sw=distance(p[11],p[12]);
  if (sw<20 || t.length<30) return null;
  const chest=band(m,t.y(0.33),t.x(0.33));
  const shoulder=band(m,t.y(-0.03),t.x(-0.03));
  let waist: [number,number] | null=null, hip: [number,number] | null=null;
  for (let f=0.5;f<0.9;f+=0.02) { const w=band(m,t.y(f),t.x(f)); if (w && (!waist || w<waist[0])) waist=[w,f]; }
  const knee=(p[25].y+p[26].y)/2, ankle=(p[27].y+p[28].y)/2;
  for (let f=0.85;f<1+0.35*(knee-t.hip.y)/t.length;f+=0.02) { const w=band(m,t.y(f),t.x(f)); if (w && (!hip || w>hip[0])) hip=[w,f]; }
  if (!chest || !waist || !hip || !shoulder || chest>1.3*sw || waist[0]>1.3*sw || hip[0]>1.6*sw) return null;
  let crotch: number | null=null;
  const start=Math.floor((knee+ankle)/2);
  if (!inside(m,t.hip.x,start)) for (let y=start;y>t.hip.y-0.3*t.length;y--) if (inside(m,t.hip.x,y)) { crotch=y; break; }
  const thighs=[perpendicular(m,p[23],p[25],0.2),perpendicular(m,p[24],p[26],0.2)].filter((v):v is number=>v!==null && v<0.6*hip![0]);
  return { stature:(m.bottom-m.top+1)*vc/sw, shoulder:shoulder/sw, chest:chest/sw, waist:waist[0]/sw, hip:hip[0]/sw, waistF:waist[1], hipF:hip[1], thigh:thighs.length?median(thighs)/sw:null, inseam:crotch!==null?(m.bottom-crotch)*vc/sw:null };
}
export function measureSide(p: Point[], m: Mask, front: Sample, vc: number): Sample | null {
  const t=torso(p), stature=(m.bottom-m.top+1)*vc;
  if (t.length<30 || !front.waistF || !front.hipF) return null;
  function scan(from: number,to: number,max: boolean) {
    const values: number[]=[];
    for (let f=from;f<to;f+=0.02) { const w=band(m,t.y(f),t.x(f)); if (w) values.push(w); }
    return values.length ? (max?Math.max(...values):Math.min(...values))/stature : null;
  }
  const bust=scan(0.15,0.55,true), waist=scan(front.waistF-0.08,front.waistF+0.08,false), hip=scan(front.hipF-0.1,front.hipF+0.15,true);
  return bust && waist && hip ? { bust,waist,hip } : null;
}
export function ellipsePerimeter(width: number, depth: number): number {
  const a=width/2,b=depth/2;
  return Math.PI*(3*(a+b)-Math.sqrt((3*a+b)*(a+3*b)));
}
export function makeProfile(front: Sample, side: Sample, heightCm: number): BodyProfile {
  if (!(heightCm>=100 && heightCm<=250) || !front.stature) throw new Error('A valid height and front scan are required.');
  const scale=heightCm/front.stature;
  const measurementsCm: BodyProfile['measurementsCm']={};
  for (const [zone,key] of [['bust','chest'],['waist','waist'],['hip','hip']] as const) {
    if (!front[key] || !side[zone]) throw new Error('Both front and side measurements are required.');
    measurementsCm[zone]=ellipsePerimeter(front[key]*scale,side[zone]*heightCm);
  }
  if (front.shoulder) measurementsCm.shoulder=front.shoulder*scale;
  if (front.inseam) measurementsCm.inseam=front.inseam*scale;
  if (front.thigh) measurementsCm.thigh=ellipsePerimeter(front.thigh*scale,front.thigh*scale*0.95);
  for (const [key,value] of Object.entries(measurementsCm)) {
    if (!Number.isFinite(value) || value<=0 || value>400) throw new Error(`Invalid ${key} estimate. Please rescan.`);
    measurementsCm[key as Zone]=Math.round(value*10)/10;
  }
  return { heightCm,measurementsCm,confidence:Object.fromEntries(Object.keys(measurementsCm).map(k=>[k,'low'])),scannedAt:new Date().toISOString(),source:'webcam',notes:['Experimental estimates; not yet validated against a tape measure.','Height is user-entered calibration; bust, waist and hip use front widths, side depths and an elliptical approximation.','Thigh uses an assumed depth ratio of 0.95. Inseam may be unavailable.','Loose clothing, posture, camera tilt, lighting and segmentation errors affect results.'] };
}
