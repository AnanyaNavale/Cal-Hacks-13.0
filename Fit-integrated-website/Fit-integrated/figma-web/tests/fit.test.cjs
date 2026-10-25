const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const ts=require('typescript');
function load(name){const m={exports:{}};const code=ts.transpileModule(fs.readFileSync(path.join(__dirname,'../src',name),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText;new Function('exports','module',code)(m.exports,m);return m.exports;}
const body=load('bodyMeasurements.ts');
const fit=load('fitEngine.ts');
test('mask preserves camera aspect and chooses largest connected person',()=>{
 const raw=new Float32Array(10000);for(let y=5;y<95;y++)for(let x=35;x<65;x++)raw[y*100+x]=1;raw[0]=1;
 const m=body.personMask(raw,100,100,200,100);assert.equal(m.width,200);assert.equal(m.height,100);assert.equal(m.top,5);assert.equal(m.bottom,94);assert.equal(m.data[0],0);
 assert.equal(body.personMask(new Float32Array(10000),100,100,100,100),null);
});
test('front and side calibration produces ellipse circumferences, not joint distances',()=>{
 const front={stature:4,shoulder:1.1,chest:1,waist:.8,hip:1.2,waistF:.7,hipF:1.05,thigh:.4,inseam:1.8};
 const p=body.makeProfile(front,{bust:.15,waist:.12,hip:.16},160);
 assert.equal(p.measurementsCm.shoulder,44);assert.equal(p.measurementsCm.inseam,72);
 assert.equal(p.measurementsCm.bust,Math.round(body.ellipsePerimeter(40,24)*10)/10);
 assert.equal(p.confidence.thigh,'low');assert.throws(()=>body.makeProfile(front,{},160));assert.throws(()=>body.makeProfile(front,{bust:.15,waist:.12,hip:.16},0));
 assert.ok(Math.abs(body.ellipsePerimeter(20,20)-20*Math.PI)<1e-9);
});
test('off-frame landmarks are rejected even with high visibility',()=>{
 const points=Array.from({length:33},()=>({x:.5,y:.5,visibility:1}));points[0].x=-.1;
 assert.match(body.framing(points,{height:100,width:100,top:5,bottom:94,data:new Uint8Array(10000)},'front'),/visible/);
});
test('chart parser rejects duplicate labels, malformed headers and nonnumeric values',()=>{
 for(const chart of ['size,bust\nM,100\nM,120','size,weight\nM,100','size,bust\nM,NaN','size,bust\nM,-5','size,bust\nM,'])assert.throws(()=>fit.parseChart(chart));
 assert.equal(fit.parseChart('size,bust,waist,hip\nM,106,86,110').length,1);
});
test('recommendation and colors change with actual selected size',()=>{
 const garment={title:'Test',fabric:'',sourceUrl:'',sizes:fit.parseChart('size,bust,waist,hip\nS,95,75,100\nM,106,86,110\nL,120,100,124')};
 const results=fit.computeFit({bust:100,waist:80,hip:104},garment);
 assert.equal(results[0].zones[0].status,'tight');assert.equal(results[1].zones[0].status,'comfortable');assert.equal(results[2].zones[0].status,'loose');assert.equal(fit.recommendation(results).size,'M');
 assert.match(fit.answerFit('waist','Could I sit?',results[1]),/6 cm/);assert.match(fit.answerFit('waist','Could I sit?',results[1]),/does not establish/);
});
test('missing charts cannot produce a recommendation and missing zones stay unknown',()=>{
 const garment={title:'Test',fabric:'',sourceUrl:'',sizes:[{label:'S',measurementsCm:{waist:86}},{label:'M',measurementsCm:{bust:106}}]};
 const results=fit.computeFit({bust:100,waist:80},garment);assert.equal(fit.recommendation(results),null);assert.equal(results[0].zones[0].status,'unknown');
});
