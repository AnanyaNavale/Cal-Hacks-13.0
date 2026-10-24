import { Text } from '@react-three/drei';
import { Layer, LayerUpdate } from './types';

interface StackLayerProps {
  layer: Layer;
  index: number;
  update: LayerUpdate | null;
}

export default function StackLayer({ layer, index, update }: StackLayerProps) {
  // Staggers the plates evenly along the vertical Y-axis
  const yPosition = (6 / 2 - index) * 1.5;
  const isGlowing = update?.status === 'changed' || update?.status === 'new';
  const hasAuthFlag = update?.flags.includes('touches_auth');

  // Change plate colors based on the incoming stream event pipeline status
  let layerColor = '#444466'; // Default translucent slate plate
  if (isGlowing) layerColor = '#3b82f6'; // Bright neon blue pulse on code change
  if (hasAuthFlag) layerColor = '#ef4444'; // Red alert warning for unreviewed auth changes

  return (
    <group position={[0, yPosition, 0]}>
      {/* 3D Glass Layer Plate */}
      <mesh>
        <boxGeometry args={[4, 0.1, 2.5]} />
        <meshStandardMaterial 
          color={layerColor} 
          transparent 
          opacity={0.65} 
          roughness={0.2} 
        />
      </mesh>

      {/* Text label locked onto the left edge of the layer */}
      <Text position={[-2.5, 0, 0]} fontSize={0.25} color="#ffffff" anchorX="right">
        {layer.toUpperCase()}
      </Text>

      {/* Floating sphere alert badge if a risky action is caught */}
      {update && update.flags.length > 0 && (
        <mesh position={[1.8, 0.3, 0]}>
          <sphereGeometry args={[0.15, 16, 16]} />
          <meshBasicMaterial color={hasAuthFlag ? '#ef4444' : '#f59e0b'} />
        </mesh>
      )}
    </group>
  );
}