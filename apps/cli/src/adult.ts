// The same model as the browser, run by onnxruntime-node (CPU).
import * as ort from 'onnxruntime-node';
import { S, DROP_C, type RunModel } from '@flymorph/core';

export async function loadModel(path: string): Promise<RunModel> {
  const session = await ort.InferenceSession.create(path);
  return async (image, drop, T) => {
    const out = await session.run({
      image: new ort.Tensor('float32', image, [T, 3, S, S]),
      drop: new ort.Tensor('float32', drop, [T, DROP_C, 1, 1]),
    });
    return out.probs.data as Float32Array;
  };
}
