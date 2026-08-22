import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import {
  Aperture, Camera, Check, CircleHelp, Film, Gauge,
  Info, Layers3, Loader2, Pause, Play, Radio, RotateCcw, ScanLine,
  Settings2, SlidersHorizontal, Sparkles, Square, Target, Upload, Video,
  X, Zap,
} from 'lucide-react';

type SourceKind = 'none' | 'camera' | 'file';
type Status = 'idle' | 'loading' | 'ready' | 'error';
type RawDetection = { class: string; score: number; bbox: [number, number, number, number] };
type Detection = RawDetection & { trackId?: number };
type Detector = { detect: (video: HTMLVideoElement) => Promise<RawDetection[]> };
type Track = { id: number; x: number; y: number; seen: number };

declare global {
  interface Window {
    tf?: unknown;
    cocoSsd?: { load: (options?: { base?: string }) => Promise<Detector> };
  }
}

const queryClient = new QueryClient();
const modelUrls = [
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js',
  'https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js',
];

function loadScript(src: string) {
  return new Promise<void>((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) { resolve(); return; }
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('The detector could not be reached. Check your connection and try again.'));
    document.head.appendChild(script);
  });
}

function centerOf(box: RawDetection['bbox']) {
  return { x: box[0] + box[2] / 2, y: box[1] + box[3] / 2 };
}

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds) || seconds <= 0) return '00:00';
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectorRef = useRef<Detector | null>(null);
  const tracksRef = useRef<Track[]>([]);
  const nextTrackRef = useRef(1);
  const frameRef = useRef<number | null>(null);
  const inferBusyRef = useRef(false);
  const lastInferRef = useRef(0);
  const fpsRef = useRef({ frames: 0, started: performance.now(), value: 0 });

  const [source, setSource] = useState<SourceKind>('none');
  const [modelStatus, setModelStatus] = useState<Status>('idle');
  const [modelError, setModelError] = useState('');
  const [sourceError, setSourceError] = useState('');
  const [ready, setReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [detectionEnabled, setDetectionEnabled] = useState(true);
  const [trackerEnabled, setTrackerEnabled] = useState(true);
  const [threshold, setThreshold] = useState(0.55);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [sessionStarted] = useState(Date.now());
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  const loadModel = useCallback(async () => {
    if (detectorRef.current || modelStatus === 'loading') return;
    setModelStatus('loading');
    setModelError('');
    try {
      for (const url of modelUrls) await loadScript(url);
      if (!window.cocoSsd) throw new Error('COCO-SSD did not initialize.');
      detectorRef.current = await window.cocoSsd.load({ base: 'mobilenet_v2' });
      setModelStatus('ready');
    } catch (error) {
      setModelStatus('error');
      setModelError(error instanceof Error ? error.message : 'Unable to load the in-browser model.');
    }
  }, [modelStatus]);

  useEffect(() => { void loadModel(); }, [loadModel]);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsPlaying(false);
  }, []);

  const startCamera = useCallback(async () => {
    setSourceError('');
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access is not supported in this browser.');
      stopCamera();
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      streamRef.current = stream;
      setSource('camera');
      setReady(true);
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play();
        setIsPlaying(true);
      }
    } catch (error) {
      setSource('none');
      setReady(false);
      setSourceError(error instanceof Error ? error.message : 'Camera permission was not granted.');
    }
  }, [stopCamera]);

  const chooseFile = useCallback((file?: File) => {
    if (!file || !file.type.startsWith('video/')) {
      setSourceError('Choose a video file in MP4, WebM, or another browser-supported format.');
      return;
    }
    stopCamera();
    setSourceError('');
    setDetections([]);
    setSource('file');
    setReady(false);
    setIsPlaying(false);
    const video = videoRef.current;
    if (video) {
      video.srcObject = null;
      video.src = URL.createObjectURL(file);
      video.load();
    }
  }, [stopCamera]);

  const togglePlayback = useCallback(async () => {
    const video = videoRef.current;
    if (!video || !ready) return;
    if (video.paused) { await video.play(); setIsPlaying(true); }
    else { video.pause(); setIsPlaying(false); }
  }, [ready]);

  const assignTracks = useCallback((items: RawDetection[]) => {
    if (!trackerEnabled) return items;
    const now = Date.now();
    const tracks = tracksRef.current;
    const used = new Set<number>();
    return items.map((item) => {
      const center = centerOf(item.bbox);
      let best: Track | undefined;
      let bestDistance = 105;
      tracks.forEach((track) => {
        const distance = Math.hypot(track.x - center.x, track.y - center.y);
        if (!used.has(track.id) && distance < bestDistance) { best = track; bestDistance = distance; }
      });
      if (!best) {
        best = { id: nextTrackRef.current++, x: center.x, y: center.y, seen: now };
        tracks.push(best);
      } else {
        best.x = center.x; best.y = center.y; best.seen = now; used.add(best.id);
      }
      return { ...item, trackId: best.id };
    });
  }, [trackerEnabled]);

  const drawScene = useCallback((items: Detection[]) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) return;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const context = canvas.getContext('2d');
    if (!context) return;
    context.drawImage(video, 0, 0, width, height);
    context.lineWidth = Math.max(2, width / 420);
    context.font = `600 ${Math.max(12, width / 58)}px Manrope, sans-serif`;
    items.forEach((item) => {
      const [x, y, w, h] = item.bbox;
      const color = item.trackId ? '#f2a23a' : '#32b698';
      context.strokeStyle = color;
      context.strokeRect(x, y, w, h);
      const label = `${item.class}${item.trackId ? `  #${item.trackId}` : ''}`;
      const labelWidth = context.measureText(label).width + 18;
      context.fillStyle = color;
      context.fillRect(x, Math.max(0, y - 26), labelWidth, 26);
      context.fillStyle = '#10282c';
      context.fillText(label, x + 9, Math.max(18, y - 8));
    });
  }, []);

  const infer = useCallback(async () => {
    const video = videoRef.current;
    const detector = detectorRef.current;
    if (!video || !detector || video.readyState < 2 || inferBusyRef.current) return;
    inferBusyRef.current = true;
    try {
      const raw = await detector.detect(video);
      const filtered = raw.filter((item) => item.score >= threshold);
      const tracked = assignTracks(filtered);
      setDetections(tracked);
      setFrameCount((value) => value + 1);
    } finally {
      inferBusyRef.current = false;
    }
  }, [assignTracks, threshold]);

  useEffect(() => {
    if (!ready) return;
    const draw = (time: number) => {
      const video = videoRef.current;
      if (video && video.readyState >= 2) {
        if (isPlaying && detectionEnabled && detectorRef.current && time - lastInferRef.current > 125) {
          lastInferRef.current = time;
          void infer();
        }
        drawScene(detections);
        fpsRef.current.frames += 1;
        const elapsed = time - fpsRef.current.started;
        if (elapsed > 1000) {
          setFps(Math.round((fpsRef.current.frames * 1000) / elapsed));
          fpsRef.current = { frames: 0, started: time, value: 0 };
        }
        setCurrentTime(video.currentTime);
      }
      frameRef.current = requestAnimationFrame(draw);
    };
    frameRef.current = requestAnimationFrame(draw);
    return () => { if (frameRef.current) cancelAnimationFrame(frameRef.current); };
  }, [ready, isPlaying, detectionEnabled, infer, drawScene, detections]);

  useEffect(() => () => {
    stopCamera();
    if (videoRef.current?.src?.startsWith('blob:')) URL.revokeObjectURL(videoRef.current.src);
  }, [stopCamera]);

  const onLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video) return;
    setDuration(video.duration || 0);
    setReady(true);
    setIsPlaying(false);
  };

  const captureFrame = () => {
    const canvas = canvasRef.current;
    if (!canvas || !ready) return;
    const link = document.createElement('a');
    link.download = `tracking-frame-${new Date().toISOString().slice(0, 19).replaceAll(':', '-')}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  const resetStudio = () => {
    stopCamera();
    const video = videoRef.current;
    if (video?.src?.startsWith('blob:')) URL.revokeObjectURL(video.src);
    if (video) { video.removeAttribute('src'); video.load(); }
    setSource('none'); setReady(false); setDetections([]); setSourceError(''); setCurrentTime(0); setDuration(0);
    tracksRef.current = []; setFrameCount(0);
  };

  const classes = useMemo(() => {
    const counts = new Map<string, number>();
    detections.forEach((item) => counts.set(item.class, (counts.get(item.class) ?? 0) + 1));
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  }, [detections]);
  const sessionMinutes = Math.max(1, Math.round((Date.now() - sessionStarted) / 60000));

  return (
    <div className="grain min-h-[100dvh] bg-background text-foreground">
      <header className="border-b border-border/80 bg-card/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between px-4 py-4 sm:px-7">
          <div className="flex items-center gap-3">
            <div className="grid size-10 place-items-center rounded-xl bg-primary text-primary-foreground shadow-sm">
              <ScanLine size={21} strokeWidth={2.3} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-[15px] font-extrabold tracking-[-.02em]">Object Tracking Studio</h1>
                <span className="hidden rounded-full bg-secondary px-2 py-0.5 font-mono text-[9px] font-medium uppercase tracking-[.12em] text-muted-foreground sm:inline">Lab / 01</span>
              </div>
              <p className="font-mono text-[10px] uppercase tracking-[.1em] text-muted-foreground">Seeing motion become data</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className={`hidden items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] font-semibold sm:flex ${modelStatus === 'ready' ? 'border-primary/25 bg-primary/8 text-primary' : modelStatus === 'error' ? 'border-destructive/30 bg-destructive/8 text-destructive' : 'border-border bg-muted/50 text-muted-foreground'}`}>
              <span className={`size-1.5 rounded-full ${modelStatus === 'ready' ? 'bg-primary pulse-dot' : modelStatus === 'error' ? 'bg-destructive' : 'bg-accent'}`} />
              {modelStatus === 'ready' ? 'Model ready' : modelStatus === 'loading' ? 'Loading model' : modelStatus === 'error' ? 'Model unavailable' : 'Model idle'}
            </div>
            <button type="button" onClick={() => void loadModel()} disabled={modelStatus === 'loading'} data-testid="button-reload-model" className="rounded-lg p-2 text-muted-foreground transition hover:bg-muted hover:text-foreground disabled:opacity-50" aria-label="Reload detector">
              <RotateCcw size={17} className={modelStatus === 'loading' ? 'animate-spin' : ''} />
            </button>
            <button type="button" data-testid="button-help" className="rounded-lg p-2 text-muted-foreground transition hover:bg-muted hover:text-foreground" aria-label="Help"><CircleHelp size={18} /></button>
          </div>
        </div>
      </header>

      <main className="studio-grid mx-auto grid max-w-[1500px] gap-5 px-4 py-5 sm:px-7 lg:grid-cols-[minmax(0,1fr)_350px] lg:py-7">
        <section className="min-w-0 space-y-5">
          <div className="flex flex-wrap items-end justify-between gap-3 animate-rise">
            <div>
              <p className="mb-1 font-mono text-[10px] uppercase tracking-[.18em] text-primary">Workspace / Live canvas</p>
              <h2 className="text-2xl font-extrabold tracking-[-.04em] sm:text-3xl">Motion, measured.</h2>
            </div>
            <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
              <span className="font-mono">threshold</span>
              <span className="rounded-md bg-card px-2 py-1 font-mono font-medium text-foreground">{Math.round(threshold * 100)}%</span>
            </div>
          </div>

          <div className="overflow-hidden rounded-2xl border border-card-border bg-card shadow-md animate-rise delay-1">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border bg-card px-4 py-3 sm:px-5">
              <div className="flex items-center gap-2">
                <span className={`size-2 rounded-full ${ready && isPlaying ? 'bg-primary pulse-dot' : 'bg-muted-foreground/40'}`} />
                <span className="font-mono text-[10px] uppercase tracking-[.16em] text-muted-foreground">{source === 'camera' ? 'Webcam feed' : source === 'file' ? 'Local video' : 'No source selected'}</span>
              </div>
              <div className="flex items-center gap-2 font-mono text-[10px] text-muted-foreground">
                <span>{ready ? `${videoRef.current?.videoWidth ?? 0} × ${videoRef.current?.videoHeight ?? 0}` : '— × —'}</span>
                <span className="text-border">|</span>
                <span data-testid="text-fps">{fps} fps</span>
              </div>
            </div>
            <div className="video-shell relative overflow-hidden bg-[#18373b]">
              <canvas ref={canvasRef} data-testid="canvas-video-output" className={`absolute inset-0 size-full object-contain ${ready ? 'opacity-100' : 'opacity-0'}`} />
              <video ref={videoRef} onLoadedMetadata={onLoadedMetadata} onEnded={() => setIsPlaying(false)} playsInline muted className="pointer-events-none absolute inset-0 size-full object-contain opacity-0" />
              {!ready && (
                <div className="absolute inset-0 grid place-items-center px-6 text-center">
                  <div className="max-w-sm">
                    <div className="relative mx-auto mb-5 grid size-16 place-items-center rounded-2xl border border-primary/35 bg-primary/10 text-primary">
                      <Target size={28} strokeWidth={1.5} />
                      <span className="scan-line absolute left-2 right-2 top-0 h-px bg-accent" />
                    </div>
                    <h3 className="text-lg font-bold tracking-[-.025em] text-[#eef2e8]">{sourceError ? 'Source needs attention' : 'Give the model something to see'}</h3>
                    <p className="mt-2 text-sm leading-6 text-[#adc5bd]">{sourceError || 'Enable your webcam or bring in a local video. Detections will appear here once the model has a frame.'}</p>
                    <div className="mt-5 flex flex-wrap justify-center gap-2">
                      <button type="button" onClick={() => void startCamera()} data-testid="button-start-camera-empty" className="inline-flex items-center gap-2 rounded-lg bg-primary px-3.5 py-2.5 text-xs font-bold text-primary-foreground transition hover:-translate-y-0.5 hover:shadow-lg"><Camera size={15} /> Enable webcam</button>
                      <button type="button" onClick={() => fileInputRef.current?.click()} data-testid="button-upload-empty" className="inline-flex items-center gap-2 rounded-lg border border-[#adc5bd]/30 bg-[#25484a] px-3.5 py-2.5 text-xs font-bold text-[#eef2e8] transition hover:bg-[#2b5657]"><Upload size={15} /> Upload video</button>
                    </div>
                  </div>
                </div>
              )}
              {ready && detectionEnabled && modelStatus === 'loading' && (
                <div className="absolute left-4 top-4 flex items-center gap-2 rounded-md bg-[#10282c]/90 px-3 py-2 font-mono text-[10px] text-[#d4e2d9]"><Loader2 size={13} className="animate-spin text-accent" /> INITIALIZING DETECTOR</div>
              )}
              {ready && detectionEnabled && modelStatus === 'error' && (
                <div className="absolute left-4 top-4 flex items-center gap-2 rounded-md bg-destructive/90 px-3 py-2 text-[10px] font-bold text-destructive-foreground"><Info size={13} /> Detector unavailable</div>
              )}
              {ready && detectionEnabled && modelStatus === 'ready' && !detections.length && isPlaying && (
                <div className="absolute bottom-4 left-4 flex items-center gap-2 rounded-md bg-[#10282c]/85 px-3 py-2 font-mono text-[10px] text-[#d4e2d9]"><span className="size-1.5 rounded-full bg-accent pulse-dot" /> SCANNING FRAME</div>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-3 border-t border-border bg-muted/35 px-4 py-3 sm:px-5">
              <button type="button" onClick={togglePlayback} disabled={!ready} data-testid="button-toggle-playback" className="grid size-9 place-items-center rounded-lg bg-foreground text-background transition hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-30" aria-label={isPlaying ? 'Pause video' : 'Play video'}>{isPlaying ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}</button>
              {source === 'file' && <div className="min-w-[140px] flex-1"><input type="range" min="0" max={duration || 1} step=".01" value={currentTime} onChange={(event) => { if (videoRef.current) videoRef.current.currentTime = Number(event.target.value); setCurrentTime(Number(event.target.value)); }} disabled={!ready} data-testid="input-video-scrubber" className="h-1.5 w-full cursor-pointer" /><div className="mt-1 flex justify-between font-mono text-[9px] text-muted-foreground"><span>{formatTime(currentTime)}</span><span>{formatTime(duration)}</span></div></div>}
              <div className="ml-auto flex items-center gap-1">
                <button type="button" onClick={captureFrame} disabled={!ready} data-testid="button-capture-frame" className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-xs font-bold transition hover:border-primary/40 hover:text-primary disabled:opacity-35"><Aperture size={15} /> <span className="hidden sm:inline">Capture frame</span></button>
                <button type="button" onClick={resetStudio} disabled={!ready && source === 'none'} data-testid="button-clear-source" className="rounded-lg p-2 text-muted-foreground transition hover:bg-card hover:text-destructive disabled:opacity-35" aria-label="Clear source"><X size={16} /></button>
              </div>
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-3 animate-rise delay-2">
            <Metric label="Objects in frame" value={String(detections.length)} detail={detections.length ? 'live detections' : 'awaiting frame'} icon={<Layers3 size={16} />} />
            <Metric label="Tracked identities" value={String(new Set(detections.map((d) => d.trackId).filter(Boolean)).size)} detail={trackerEnabled ? 'centroid matching' : 'tracker paused'} icon={<Target size={16} />} />
            <Metric label="Session frames" value={frameCount.toLocaleString()} detail={`${sessionMinutes} min session`} icon={<Film size={16} />} />
          </div>
        </section>

        <aside className="space-y-5">
          <div className="rounded-2xl border border-card-border bg-card p-4 shadow-sm animate-rise delay-1 sm:p-5">
            <div className="mb-4 flex items-center justify-between">
              <div><p className="font-mono text-[10px] uppercase tracking-[.16em] text-primary">Input</p><h3 className="mt-1 text-base font-bold tracking-[-.02em]">Choose a source</h3></div>
              <Settings2 size={17} className="text-muted-foreground" />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <SourceButton active={source === 'camera'} icon={<Camera size={16} />} label="Webcam" onClick={() => void startCamera()} testId="button-source-webcam" />
              <SourceButton active={source === 'file'} icon={<Upload size={16} />} label="Local video" onClick={() => fileInputRef.current?.click()} testId="button-source-file" />
            </div>
            <input ref={fileInputRef} type="file" accept="video/*" onChange={(event) => chooseFile(event.target.files?.[0])} className="hidden" data-testid="input-video-file" />
            {source === 'camera' && <button type="button" onClick={stopCamera} data-testid="button-stop-camera" className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg border border-destructive/25 bg-destructive/5 py-2.5 text-xs font-bold text-destructive transition hover:bg-destructive/10"><Square size={13} fill="currentColor" /> Stop camera</button>}
            {source === 'file' && <p className="mt-3 flex items-center gap-2 text-[11px] leading-5 text-muted-foreground"><Video size={14} className="shrink-0 text-primary" /> Video stays in your browser. Nothing is uploaded.</p>}
            {modelStatus === 'error' && <div className="mt-4 rounded-lg border border-destructive/20 bg-destructive/5 p-3 text-[11px] leading-5 text-destructive"><strong className="block">Model error</strong>{modelError}<button type="button" onClick={() => void loadModel()} data-testid="button-retry-model" className="mt-2 font-bold underline">Try again</button></div>}
          </div>

          <div className="rounded-2xl border border-card-border bg-card p-4 shadow-sm animate-rise delay-2 sm:p-5">
            <div className="mb-5 flex items-center justify-between"><div><p className="font-mono text-[10px] uppercase tracking-[.16em] text-primary">Pipeline</p><h3 className="mt-1 text-base font-bold tracking-[-.02em]">Detection controls</h3></div><SlidersHorizontal size={17} className="text-muted-foreground" /></div>
            <ControlRow label="Object detection" hint={detectionEnabled ? 'running' : 'paused'} on={detectionEnabled} onToggle={() => setDetectionEnabled((value) => !value)} testId="switch-detection" />
            <div className="my-5 h-px bg-border" />
            <div>
              <div className="mb-2 flex items-center justify-between"><label htmlFor="confidence" className="text-xs font-bold">Confidence threshold</label><span className="rounded-md bg-secondary px-2 py-1 font-mono text-[10px] font-medium text-secondary-foreground">{Math.round(threshold * 100)}%</span></div>
              <input id="confidence" type="range" min=".2" max=".9" step=".05" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} data-testid="input-confidence-threshold" className="w-full cursor-pointer" />
              <div className="mt-1 flex justify-between font-mono text-[9px] text-muted-foreground"><span>more objects</span><span>more certain</span></div>
            </div>
            <div className="my-5 h-px bg-border" />
            <ControlRow label="Persistent IDs" hint={trackerEnabled ? 'centroid tracker' : 'off'} on={trackerEnabled} onToggle={() => setTrackerEnabled((value) => !value)} testId="switch-tracker" />
          </div>

          <div className="rounded-2xl border border-card-border bg-card p-4 shadow-sm animate-rise delay-3 sm:p-5">
            <div className="mb-4 flex items-center justify-between"><div><p className="font-mono text-[10px] uppercase tracking-[.16em] text-primary">Telemetry</p><h3 className="mt-1 text-base font-bold tracking-[-.02em]">What the model sees</h3></div><Gauge size={17} className="text-muted-foreground" /></div>
            {classes.length ? <div className="space-y-2">{classes.map(([label, count]) => <div key={label} className="flex items-center justify-between rounded-lg bg-muted/60 px-3 py-2.5 text-xs"><span className="font-semibold capitalize">{label}</span><span className="font-mono text-muted-foreground">{count} visible</span></div>)}</div> : <div className="rounded-xl border border-dashed border-border bg-muted/35 px-4 py-6 text-center"><Sparkles size={18} className="mx-auto text-accent" /><p className="mt-2 text-xs font-bold">No detections yet</p><p className="mt-1 text-[11px] leading-5 text-muted-foreground">{ready ? 'Play the source or try lowering the threshold.' : 'Start a source to populate live class metrics.'}</p></div>}
            <div className="mt-4 grid grid-cols-2 gap-2"><MiniStat label="Frame rate" value={`${fps} fps`} /><MiniStat label="Confidence" value={detections.length ? `${Math.round(Math.max(...detections.map((d) => d.score)) * 100)}% max` : '—'} /></div>
          </div>
        </aside>
      </main>
      <footer className="mx-auto flex max-w-[1500px] items-center justify-between px-4 pb-6 pt-1 font-mono text-[9px] uppercase tracking-[.12em] text-muted-foreground sm:px-7"><span>In-browser inference · COCO-SSD</span><span className="hidden sm:inline">Local by default / v1.0</span></footer>
    </div>
  );
}

function Metric({ label, value, detail, icon }: { label: string; value: string; detail: string; icon: ReactNode }) {
  return <div className="rounded-xl border border-card-border bg-card p-4 shadow-sm"><div className="flex items-start justify-between"><span className="text-muted-foreground">{icon}</span><span className="font-mono text-[9px] uppercase tracking-[.12em] text-muted-foreground">live</span></div><div className="mt-3 text-2xl font-extrabold tracking-[-.05em]">{value}</div><div className="mt-1 text-xs font-bold">{label}</div><div className="mt-1 font-mono text-[9px] text-muted-foreground">{detail}</div></div>;
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return <div className="rounded-lg bg-muted/60 px-3 py-2"><div className="font-mono text-[9px] uppercase tracking-[.1em] text-muted-foreground">{label}</div><div className="mt-1 text-xs font-bold">{value}</div></div>;
}

function SourceButton({ active, icon, label, onClick, testId }: { active: boolean; icon: ReactNode; label: string; onClick: () => void; testId: string }) {
  return <button type="button" onClick={onClick} data-testid={testId} className={`flex items-center justify-center gap-2 rounded-lg border px-3 py-3 text-xs font-bold transition ${active ? 'border-primary/40 bg-primary/10 text-primary' : 'border-border bg-muted/35 text-muted-foreground hover:border-primary/30 hover:text-foreground'}`}>{icon}{label}{active && <Check size={13} />}</button>;
}

function ControlRow({ label, hint, on, onToggle, testId }: { label: string; hint: string; on: boolean; onToggle: () => void; testId: string }) {
  return <div className="flex items-center justify-between gap-3"><div><p className="text-xs font-bold">{label}</p><p className="mt-1 font-mono text-[9px] text-muted-foreground">{hint}</p></div><button type="button" role="switch" aria-checked={on} onClick={onToggle} data-testid={testId} className={`relative h-6 w-11 rounded-full p-1 transition ${on ? 'bg-primary' : 'bg-secondary'}`}><span className={`block size-4 rounded-full bg-card shadow-sm transition-transform ${on ? 'translate-x-5' : 'translate-x-0'}`} /></button></div>;
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  return <ErrorBoundary>{children}</ErrorBoundary>;
}

export default function RootApp() {
  return <QueryClientProvider client={queryClient}><TooltipProvider><RoutedErrorBoundary><App /></RoutedErrorBoundary></TooltipProvider><Toaster /></QueryClientProvider>;
}