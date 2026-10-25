export type FitZone = 'bust'|'waist'|'hip';
export type FitStatus = 'tight'|'snug'|'comfortable'|'loose'|'unknown';
export interface Garment {title:string;sourceUrl:string;fabric:string;sizes:{label:string;measurementsCm:Partial<Record<FitZone,number>>}[]}
export interface ZoneFit {zone:FitZone;easeCm:number|null;status:FitStatus}
export interface SizeFit {size:string;zones:ZoneFit[];score:number|null}
export function parseChart(text:string):Garment['sizes'] {
  const lines=text.trim().split(/\r?\n/).filter(s=>s.trim());
  if(lines.length<2 || lines.length>31)throw new Error('Add a header and 1–30 size rows.');
  const header=lines[0].toLowerCase().split(',').map(s=>s.trim());
  if(header[0]!=='size' || header.some((s,i)=>i>0 && !['bust','waist','hip'].includes(s)) || new Set(header).size!==header.length || header.length<2)throw new Error('Use CSV headers: size,bust,waist,hip. You may omit unavailable zones.');
  const labels=new Set<string>();
  return lines.slice(1).map(line=>{
    const cells=line.split(',').map(s=>s.trim());
    if(cells.length!==header.length || !cells[0] || cells[0].length>20 || labels.has(cells[0]))throw new Error('Each row needs a unique size and matching columns.');
    labels.add(cells[0]);
    const measurementsCm:Garment['sizes'][number]['measurementsCm']={};
    for(let i=1;i<header.length;i++){
      if(!cells[i])continue;
      const value=Number(cells[i]);
      if(!Number.isFinite(value)||value<10||value>400)throw new Error('Chart values must be garment circumferences in cm, between 10 and 400.');
      measurementsCm[header[i] as FitZone]=value;
    }
    if(!Object.keys(measurementsCm).length)throw new Error('Each size needs at least one garment measurement.');
    return {label:cells[0],measurementsCm};
  });
}
export function computeFit(body:Partial<Record<FitZone,number>>,garment:Garment):SizeFit[]{
  // Basic non-stretch heuristic, not cloth simulation. Missing data never earns a score.
  const results=garment.sizes.map(size=>{
    const zones:ZoneFit[]=(['bust','waist','hip'] as const).map(zone=>{
      const b=body[zone],g=size.measurementsCm[zone];
      if(b==null || g==null)return {zone,easeCm:null,status:'unknown'};
      const easeCm=Math.round((g-b)*10)/10;
      const status:FitStatus=easeCm<0?'tight':easeCm<4?'snug':easeCm<=10?'comfortable':'loose';
      return {zone,easeCm,status};
    });
    return {size:size.label,zones,score:null as number|null};
  });
  // Rank on a common set of zones so incomplete charts cannot win by omission.
  const common=(['bust','waist','hip'] as const).filter(z=>body[z]!=null && garment.sizes.every(s=>s.measurementsCm[z]!=null));
  for(const r of results)if(common.length)r.score=-r.zones.filter(z=>common.includes(z.zone)).reduce((sum,z)=>sum+Math.abs(z.easeCm!-6)+(z.easeCm!<0?30:0),0);
  return results;
}
export function recommendation(results:SizeFit[]):SizeFit|null{
  const eligible=results.filter(r=>r.score!==null && !r.zones.some(z=>z.status==='tight'));
  return eligible.sort((a,b)=>b.score!-a.score!)[0]??null;
}
export function answerFit(area:string,question:string,result:SizeFit|null):string {
  if(!question.trim())return 'Enter a question first.';
  const zone=area==='hips'?'hip':area;
  const fit=result?.zones.find(z=>z.zone===zone);
  if(!fit || fit.easeCm===null)return 'There is not enough measurement data for this area. Add a garment chart or retry the scan.';
  const base=`For size ${result!.size}, the ${area} area has ${fit.easeCm} cm of ease (garment minus body), suggesting a ${fit.status} fit under our basic non-stretch rule.`;
  return /gap|back|cling|sit|move|stretch|fabric/i.test(question)?base+' This does not establish back gaping, movement or fabric behavior; those need cut, stretch and more detailed shape information.':base+' Scan error and garment construction may change how it feels. This is a calculated explanation, not an AI answer.';
}
