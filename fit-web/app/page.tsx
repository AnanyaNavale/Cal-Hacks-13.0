"use client";

import { useEffect, useRef, useState } from "react";
import type { PoseLandmarker } from "@mediapipe/tasks-vision";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const modelRef = useRef<PoseLandmarker | null>(null);
  const animationRef = useRef<number | null>(null);
  const sessionRef = useRef(0);

  const [starting, setStarting] = useState(false);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Ready to start.");
  const [error, setError] = useState("");

  function releaseResources() {
    if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    modelRef.current?.close();
    modelRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    const canvas = canvasRef.current;
    canvas
      ?.getContext("2d")
      ?.clearRect(0, 0, canvas.width, canvas.height);
  }

  useEffect(() => {
    return () => {
      sessionRef.current += 1;

      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
      }

      streamRef.current?.getTracks().forEach((track) => track.stop());
      modelRef.current?.close();
    };
  }, []);

  function stop() {
    sessionRef.current += 1;
    releaseResources();
    setStarting(false);
    setRunning(false);
    setStatus("Camera is off.");
    setError("");
  }

  async function start() {
    const session = ++sessionRef.current;

    setStarting(true);
    setError("");
    setStatus("Loading body detection. The first load may take a moment.");

    try {
      const { FilesetResolver, PoseLandmarker, DrawingUtils } =
        await import("@mediapipe/tasks-vision");

      const files = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm"
      );

      if (session !== sessionRef.current) return;

      const model = await PoseLandmarker.createFromOptions(files, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.6,
        minPosePresenceConfidence: 0.6,
        minTrackingConfidence: 0.6,
      });

      if (session !== sessionRef.current) {
        model.close();
        return;
      }

      modelRef.current = model;
      setStatus("Allow camera access if your browser asks.");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      if (session !== sessionRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video || !canvas) {
        throw new Error("The camera preview is unavailable.");
      }

      video.srcObject = stream;
      await video.play();

      if (session !== sessionRef.current) return;

      const context = canvas.getContext("2d");

      if (!context) {
        throw new Error("The tracking overlay could not start.");
      }

      const drawing = new DrawingUtils(context);
      let lastVideoTime = -1;
      let lastDetectionTime = -1;

      setStarting(false);
      setRunning(true);

      function detect() {
        if (session !== sessionRef.current) return;

        try {
          const now = performance.now();

          // Process each new video frame, at most about 15 times a second.
          if (
            video!.readyState >= 2 &&
            video!.currentTime !== lastVideoTime &&
            now - lastDetectionTime >= 66
          ) {
            lastVideoTime = video!.currentTime;
            lastDetectionTime = now;

            if (
              canvas!.width !== video!.videoWidth ||
              canvas!.height !== video!.videoHeight
            ) {
              canvas!.width = video!.videoWidth;
              canvas!.height = video!.videoHeight;
            }

            context!.clearRect(0, 0, canvas!.width, canvas!.height);

            model.detectForVideo(video!, now, (result) => {
              const points = result.landmarks[0];

              if (!points) {
                setStatus("Step into view so the camera can see you.");
                return;
              }

              drawing.drawConnectors(
                points,
                PoseLandmarker.POSE_CONNECTIONS,
                { color: "#2dd4bf", lineWidth: 3 }
              );

              drawing.drawLandmarks(points, {
                color: "#ffffff",
                fillColor: "#0f766e",
                radius: 4,
              });

              const required = [
                0, 11, 12, 13, 14, 15, 16,
                23, 24, 25, 26, 27, 28,
              ];

              const fullBodyVisible = required.every((index) => {
                const point = points[index];

                return (
                  (point.visibility ?? 0) > 0.6 &&
                  point.x > 0.02 &&
                  point.x < 0.98 &&
                  point.y > 0.02 &&
                  point.y < 0.98
                );
              });

              setStatus(
                fullBodyVisible
                  ? "Body landmarks are visible. Tracking is working."
                  : "Body detected. Step back until your head, hands, and feet are in view."
              );
            });
          }

          animationRef.current = requestAnimationFrame(detect);
        } catch (err) {
          sessionRef.current += 1;
          releaseResources();
          setRunning(false);
          setStatus("Body detection stopped.");
          setError(
            err instanceof Error ? err.message : "Body detection failed."
          );
        }
      }

      animationRef.current = requestAnimationFrame(detect);
    } catch (err) {
      if (session !== sessionRef.current) return;

      releaseResources();
      setStarting(false);
      setRunning(false);
      setStatus("Could not start.");

      setError(
        err instanceof Error
          ? err.message
          : "Check camera permissions and your internet connection."
      );
    }
  }

  return (
    <main className="min-h-screen bg-stone-50 px-5 py-10 text-stone-900">
      <div className="mx-auto max-w-4xl">
        <p className="text-sm font-semibold uppercase tracking-widest text-teal-800">
          Your private fitting room
        </p>
        <h1 className="mt-3 text-5xl font-bold">Fit</h1>
        <p className="mt-4 text-lg text-stone-600">
          Find your fit, starting with your webcam.
        </p>

        <section className="mt-8 rounded-3xl bg-white p-6 shadow-sm">
          <h2 className="text-2xl font-semibold">Body tracking</h2>
          <p className="mt-2 text-stone-600">
            Put your laptop on a stable surface. Use good lighting
            and make space to step back safely.
          </p>

          <div
            className="relative mt-6 overflow-hidden rounded-2xl bg-stone-900"
            style={{ aspectRatio: "16 / 9" }}
          >
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              aria-label="Live webcam preview"
              className="absolute inset-0 h-full w-full object-fill"
              style={{ transform: "scaleX(-1)" }}
              onLoadedMetadata={(event) => {
                const video = event.currentTarget;
                const container = video.parentElement;

                if (container && video.videoWidth && video.videoHeight) {
                  container.style.aspectRatio =
                    `${video.videoWidth} / ${video.videoHeight}`;
                }
              }}
            />

            <canvas
              ref={canvasRef}
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 h-full w-full"
              style={{ transform: "scaleX(-1)" }}
            />

            {!running && (
              <div className="absolute inset-0 flex items-center justify-center p-6 text-center text-white">
                {starting ? "Preparing body tracking…" : "Camera is off"}
              </div>
            )}
          </div>

          <p role="status" className="mt-4 font-medium text-teal-800">
            {status}
          </p>

          {error && (
            <p role="alert" className="mt-3 text-sm text-red-700">
              {error}
            </p>
          )}

          <div className="mt-5 flex gap-3">
            <button
              onClick={start}
              disabled={starting || running}
              className="rounded-xl bg-teal-800 px-5 py-3 font-semibold text-white disabled:opacity-50"
            >
              {starting ? "Loading…" : "Start body tracking"}
            </button>

            <button
              onClick={stop}
              disabled={!starting && !running}
              className="rounded-xl border border-stone-300 px-5 py-3 font-semibold disabled:opacity-50"
            >
              Stop
            </button>
          </div>

          <p className="mt-6 text-sm text-stone-500">
            Detection runs in your browser. This page does not upload
            or save camera frames. Measurement calculations are not
            connected yet.
          </p>
        </section>
      </div>
    </main>
  );
}