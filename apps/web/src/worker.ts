// Runs the heavy pixel loops off the main thread so the page never freezes (≈4 s for a 10 MP image).
//   mode 'calibrate'  read the micrometer photo → Calibration
//   mode 'larva'      measure larvae with a Calibration the page already has
import { calibrateFromScale, measureLarvae, MIN_CONFIDENCE, type Calibration, type RGB } from '@flymorph/core';

export type WorkerRequest = { mode: 'calibrate'; scale: RGB } | { mode: 'larva'; img: RGB; cal: Calibration };

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const post = (msg: unknown, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(msg, transfer);
  try {
    const req = e.data;
    if (req.mode === 'calibrate') {
      const cal = calibrateFromScale(req.scale);
      if (cal.confidence < MIN_CONFIDENCE)
        return post({ ok: false, needsScale: true, error: 'Could not find the micrometer bar automatically. Click its two ends on the scale photo below.' });
      return post({ ok: true, cal });
    }
    const { measurements, labels } = measureLarvae(req.img, req.cal);
    post({ ok: true, cal: req.cal, measurements, labels }, [labels.buffer]);
  } catch (err) {
    post({ ok: false, error: (err as Error).message ?? String(err) });
  }
};
