"use client";
import { useEffect, useState } from "react";
import type { RefObject } from "react";
import type { PoseLandmarker } from "@mediapipe/tasks-vision";

export function usePoseTracking(active: boolean, videoRef: RefObject<HTMLVideoElement | null>, canvasRef: RefObject<HTMLCanvasElement | null>) {
  const [message, setMessage] = useState("Enable the camera to start tracking.");
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    let frame = 0;
    let model: PoseLandmarker | null = null;
    let lastTime = -1;
    let lastRun = -1;
    async function initialize() {
      try {
        setMessage("Loading body tracking...");
        const { FilesetResolver, PoseLandmarker, DrawingUtils } = await import("@mediapipe/tasks-vision");
        const files = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm");
        if (cancelled) return;
        const created = await PoseLandmarker.createFromOptions(files, {
          baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task" },
          runningMode: "VIDEO", numPoses: 1,
          minPoseDetectionConfidence: 0.6, minPosePresenceConfidence: 0.6, minTrackingConfidence: 0.6,
        });
        if (cancelled) { created.close(); return; }
        model = created;
        function detect() {
          if (cancelled) return;
          try {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            const context = canvas?.getContext("2d");
            const now = performance.now();
            if (video && canvas && context && video.readyState >= 2 && video.currentTime !== lastTime && now - lastRun >= 66) {
              lastTime = video.currentTime;
              lastRun = now;
              if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
              }
              context.clearRect(0, 0, canvas.width, canvas.height);
              const drawing = new DrawingUtils(context);
              created.detectForVideo(video, now, (result) => {
                const points = result.landmarks[0];
                if (!points) { setMessage("No body detected. Check framing and even lighting."); return; }
                drawing.drawConnectors(points, PoseLandmarker.POSE_CONNECTIONS, { color: "#2dd4bf", lineWidth: 3 });
                drawing.drawLandmarks(points, { color: "#ffffff", fillColor: "#0f766e", radius: 3 });
                const visible = [0, 11, 12, 15, 16, 23, 24, 27, 28].every((i) => {
                  const p = points[i];
                  return (p.visibility ?? 0) > 0.6 && p.x > 0.02 && p.x < 0.98 && p.y > 0.02 && p.y < 0.98;
                });
                setMessage(visible ? "Body landmarks visible. Measurement capture is not connected yet." : "Body detected. Bring your head, hands, and feet into view.");
              });
            }
            frame = requestAnimationFrame(detect);
          } catch (error) {
            model?.close(); model = null;
            setMessage(`Tracking stopped: ${error instanceof Error ? error.message : "Unknown error"}. Stop and restart the camera to retry.`);
          }
        }
        frame = requestAnimationFrame(detect);
      } catch (error) {
        if (!cancelled) setMessage(`Tracking unavailable: ${error instanceof Error ? error.message : "Could not load model"}`);
      }
    }
    void initialize();
    return () => {
      cancelled = true;
      cancelAnimationFrame(frame);
      model?.close();
      const canvas = canvasRef.current;
      canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
    };
  }, [active, videoRef, canvasRef]);
  return message;
}
