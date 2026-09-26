import { useEffect, useState } from 'react';
import { Measure } from './Measure';
import { Review } from './Review';

// Two pages, so a hash is all the routing needed: #/review, anything else = measure.
export function App() {
  const [hash, setHash] = useState(location.hash);
  useEffect(() => {
    const on = () => setHash(location.hash);
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return (
    <main>
      <nav>
        <h1>FlyMorph</h1>
        <a href="#/">Measure</a>
        <a href="#/review">Review labels</a>
      </nav>
      {hash === '#/review' ? <Review /> : <Measure />}
    </main>
  );
}
