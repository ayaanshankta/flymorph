import { useState } from 'react';
import type { RGB } from '@flymorph/core';
import { decode } from './decode';
import { Overlay } from './Overlay';

export function Measure() {
  const [img, setImg] = useState<RGB | null>(null);
  return (
    <section>
      <label>
        Image <input type="file" accept="image/*" onChange={async (e) => e.target.files?.[0] && setImg(await decode(e.target.files[0]))} />
      </label>
      {img && <Overlay img={img} />}
    </section>
  );
}
