import { useEffect, useState } from 'react';
import { Measure, type Mode } from './Measure';
import { Review } from './Review';

// Tabs live in the URL hash (#larvae / #adults) so a link can open straight onto one. #review is the
// label-review tool: local only (it needs ml/draft_server.py), so it is not in the menu.
const TABS: { hash: string; mode: Mode; label: string }[] = [
  { hash: '#larvae', mode: 'larva', label: 'Larvae' },
  { hash: '#adults', mode: 'adult', label: 'Adult flies' },
];

export function App() {
  const [hash, setHash] = useState(location.hash || '#larvae');
  useEffect(() => {
    const on = () => setHash(location.hash || '#larvae');
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);

  if (hash === '#review' || hash === '#/review') return <main className="wrap"><Review /></main>;
  const tab = TABS.find((t) => t.hash === hash) ?? TABS[0];

  return (
    <main className="wrap">
      <header className="top">
        <div className="brand">
          <svg className="logo" viewBox="0 0 32 32" aria-hidden="true">
            <ellipse cx="10" cy="12" rx="8" ry="4.5" transform="rotate(-25 10 12)" className="wing" />
            <ellipse cx="22" cy="12" rx="8" ry="4.5" transform="rotate(25 22 12)" className="wing" />
            <ellipse cx="16" cy="21" rx="4" ry="7" className="body" />
            <circle cx="16" cy="11.5" r="3.4" className="body" />
          </svg>
          <h1>FlyMorph</h1>
        </div>
        <nav className="tabs" aria-label="What to measure">
          {TABS.map((t) => (
            <a key={t.hash} href={t.hash} aria-current={t === tab ? 'page' : undefined}>{t.label}</a>
          ))}
        </nav>
      </header>

      {TABS.map((t) => (
        <div key={t.hash} hidden={t !== tab}><Measure mode={t.mode} /></div>
      ))}

      <footer className="foot">
        Photos are measured on your own computer and never uploaded.{' '}
        <a href="https://github.com/ayaanshankta/flymorph">Source on GitHub</a>
      </footer>

    </main>
  );
}
