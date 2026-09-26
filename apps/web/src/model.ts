// Loads flynet.onnx once and exposes it as core's RunModel. Tries the GPU (WebGPU) first, then CPU (WASM).
import * as ort from 'onnxruntime-web/webgpu';
import { S, DROP_C, type RunModel } from '@flymorph/core';

// The .wasm runtime files come from the CDN copy of the exact installed version.
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;

let loading: Promise<{ run: RunModel; backend: string }> | null = null;

export function loadModel() {
  loading ??= (async () => {
    const url = new URL('flynet.onnx', document.baseURI).href;
    for (const backend of ['webgpu', 'wasm']) {
      try {
        const session = await ort.InferenceSession.create(url, { executionProviders: [backend] });
        const run: RunModel = async (image, drop, T) => {
          const out = await session.run({
            image: new ort.Tensor('float32', image, [T, 3, S, S]),
            drop: new ort.Tensor('float32', drop, [T, DROP_C, 1, 1]),
          });
          return out.probs.data as Float32Array;
        };
        return { run, backend };
      } catch (e) {
        console.warn(`[flymorph] ${backend} backend unavailable:`, e);
      }
    }
    loading = null;
    throw new Error('Could not load the adult model (flynet.onnx).');
  })();
  return loading;
}
