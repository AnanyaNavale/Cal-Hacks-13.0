import { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrthographicCamera } from '@react-three/drei';
import { Layer, LayerUpdate } from './types';
import StackLayer from './StackLayer';

const LAYERS_ORDER: Layer[] = ['ui', 'api', 'logic', 'data', 'auth', 'infra'];

export default function App() {
  const [liveUpdates, setLiveUpdates] = useState<Record<Layer, LayerUpdate | null>>({
    ui: null, api: null, logic: null, data: null, auth: null, infra: null
  });

  // Background Simulator Engine: Fakes incoming data updates from Droid logs
  useEffect(() => {
    const simulationPool: Layer[] = ['ui', 'api', 'data', 'auth'];
    const interval = setInterval(() => {
      const activeSelection = simulationPool[Math.floor(Math.random() * simulationPool.length)];
      
      const simulatedPayload: LayerUpdate = {
        t: Date.now(),
        layer: activeSelection,
        status: 'changed',
        files: [`src/${activeSelection}/index.ts`],
        flags: activeSelection === 'auth' ? ['touches_auth'] : []
      };

      setLiveUpdates(prev => ({ ...prev, [activeSelection]: simulatedPayload }));

      // Reset the bright neon glow pulse after 1.5 seconds
      setTimeout(() => {
        setLiveUpdates(prev => {
          const context = prev[activeSelection];
          if (!context) return prev;
          return { ...prev, [activeSelection]: { ...context, status: 'unchanged' } };
        });
      }, 1500);
    }, 4000); // Changes trigger automatically every 4 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#0b0f19', margin: 0, overflow: 'hidden' }}>
      <Canvas>
        {/* FIXED POSITION VECTORS: Placed the camera at an isometric 3D perspective angle */}
        <OrthographicCamera makeDefault position={[5, 5, 5]} zoom={90} />
        <ambientLight intensity={0.8} />
        <pointLight position={[10, 10, 10]} intensity={1.2} />
        
        {/* Tilts the canvas world view matrix to display clear layer depth */}
        <group rotation={[0, -Math.PI / 4, 0]}>
          {LAYERS_ORDER.map((layer, index) => (
            <StackLayer key={layer} layer={layer} index={index} update={liveUpdates[layer]} />
          ))}
        </group>
      </Canvas>
    </div>
  );
}
