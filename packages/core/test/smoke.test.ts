import { expect, test } from 'vitest';
import { VERSION } from '../src/index';
test('core loads', () => expect(VERSION).toBe('0.1.0'));
