// Runs the heavy pixel loops off the main thread so the page never freezes (≈4 s for a 10 MP image).
import { calibrateFromScale, measureLarvae, MIN_CONFIDENCE, type Calibration, type RGB } from '@flymorph/core';

export type WorkerRequest = { img: RGB; scale: RGB | null; manualPxPerMm?: number };

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const { img, scale, manualPxPerMm } = e.data;
  const post = (msg: unknown, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(msg, transfer);
  try {
    let cal: Calibration;
    if (manualPxPerMm) cal = { pxPerMm: manualPxPerMm, confidence: Infinity, method: 'manual' };
    else if (!scale) throw new Error('Add a scale image, or click the two ends of the bar to calibrate.');
    else {
      cal = calibrateFromScale(scale);
      if (cal.confidence < MIN_CONFIDENCE)
        throw new Error('Could not find the micrometer bar automatically. Click its two ends on the scale image.');
    }
    const { measurements, labels } = measureLarvae(img, cal);
    post({ ok: true, cal, measurements, labels }, [labels.buffer]);
  } catch (err) {
    post({ ok: false, error: (err as Error).message ?? String(err) });
  }
};
