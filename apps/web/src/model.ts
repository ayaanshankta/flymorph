// Loads flynet.onnx once and exposes it as core's RunModel.
// Uses ONNX Runtime's plain WebAssembly (CPU) build, served from this site's own ort/ folder: it is 14 MB,
// works in every browser and fits static hosts with per-file size limits. (The WebGPU build is 28 MB.)
import * as ort from 'onnxruntime-web/wasm';
import { S, DROP_C, type RunModel } from '@flymorph/core';

ort.env.wasm.wasmPaths = new URL('ort/', document.baseURI).href;

let loading: Promise<RunModel> | null = null;

export function loadModel(): Promise<RunModel> {
  loading ??= (async () => {
    const session = await ort.InferenceSession.create(new URL('flynet.onnx', document.baseURI).href, {
      executionProviders: ['wasm'],
    });
    const run: RunModel = async (image, drop, T) => {
      const out = await session.run({
        image: new ort.Tensor('float32', image, [T, 3, S, S]),
        drop: new ort.Tensor('float32', drop, [T, DROP_C, 1, 1]),
      });
      return out.probs.data as Float32Array;
    };
    return run;
  })().catch((e) => {
    loading = null; // let the next click try again
    throw new Error(`Could not load the adult-fly model (${(e as Error).message}).`);
  });
  return loading;
}
