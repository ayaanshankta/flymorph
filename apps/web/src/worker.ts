// Runs the heavy pixel loops off the main thread so the page never freezes (≈4 s for a 10 MP image).
import { calibrateFromScale, measureLarvae, MIN_CONFIDENCE, type Calibration, type RGB } from '@flymorph/core';

// mode 'calibrate' only reads the scale (adult mode runs the model on the main thread, where WebGPU lives).
export type WorkerRequest = { mode: 'larva' | 'calibrate'; img: RGB; scale: RGB | null; manualPxPerMm?: number };

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const { mode, img, scale, manualPxPerMm } = e.data;
  const post = (msg: unknown, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(msg, transfer);
  try {
    let cal: Calibration;
    if (manualPxPerMm) cal = { pxPerMm: manualPxPerMm, confidence: Infinity, method: 'manual' };
    else if (!scale) return post({ ok: false, error: 'Add a scale image to calibrate.' });
    else {
      cal = calibrateFromScale(scale);
      if (cal.confidence < MIN_CONFIDENCE)
        return post({ ok: false, needsScale: true, error: 'Could not find the micrometer bar automatically. Click its two ends on the scale image.' });
    }
    if (mode === 'calibrate') return post({ ok: true, cal, measurements: [], labels: new Int32Array(0) });
    const { measurements, labels } = measureLarvae(img, cal);
    post({ ok: true, cal, measurements, labels }, [labels.buffer]);
  } catch (err) {
    post({ ok: false, error: (err as Error).message ?? String(err) });
  }
};
