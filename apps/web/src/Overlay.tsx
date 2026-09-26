import { useEffect, useRef } from 'react';
import type { RGB } from '@flymorph/core';

export type Box = { x: number; y: number; w: number; h: number; text: string };
type Props = {
  img: RGB;
  labels?: ArrayLike<number>; // one label per image pixel; 0 = nothing
  color?: (label: number) => [number, number, number] | null;
  boxes?: Box[];
  onClick?: (x: number, y: number, e: React.MouseEvent) => void; // image-pixel coordinates
};

export function Overlay({ img, labels, color, boxes = [], onClick }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current!;
    c.width = img.w;
    c.height = img.h;
    const ctx = c.getContext('2d')!;
    const px = new ImageData(new Uint8ClampedArray(img.data), img.w, img.h);
    if (labels && color)
      for (let i = 0; i < labels.length; i++) {
        const col = color(labels[i]);
        if (!col) continue;
        for (let k = 0; k < 3; k++) px.data[i * 4 + k] = px.data[i * 4 + k] * 0.5 + col[k] * 0.5;
      }
    ctx.putImageData(px, 0, 0);
    ctx.lineWidth = Math.max(2, img.w / 600);
    ctx.font = `${Math.max(14, img.w / 60)}px system-ui`;
    ctx.strokeStyle = ctx.fillStyle = '#39ff88';
    for (const b of boxes) {
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.fillText(b.text, b.x, Math.max(20, b.y - 8));
    }
  }, [img, labels, color, boxes]);

  const click = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onClick) return;
    const r = e.currentTarget.getBoundingClientRect();
    onClick(((e.clientX - r.left) / r.width) * img.w, ((e.clientY - r.top) / r.height) * img.h, e);
  };
  return <canvas ref={ref} onClick={click} className="overlay" />;
}
