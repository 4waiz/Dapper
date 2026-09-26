// Seeded pseudo-random numbers (mulberry32) with the distributions the live
// synthetic generator needs.
//
// This is NOT a port of NumPy's PCG64: the published traces cannot be
// reproduced in the browser, which is exactly why replay mode loads the exported
// Python traces and live mode is labelled "live synthetic".

export class RNG {
  constructor(seed = 1) {
    this.state = seed >>> 0;
    this.spare = null;
  }

  random() {
    let t = (this.state = (this.state + 0x6d2b79f5) | 0);
    t = Math.imul(t ^ (t >>> 15), 1 | t);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  uniform(a, b) {
    return a + (b - a) * this.random();
  }

  /** Marsaglia polar method, with the second variate kept for the next call. */
  normal(mu = 0, sd = 1) {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return mu + sd * v;
    }
    let u;
    let v;
    let s;
    do {
      u = 2 * this.random() - 1;
      v = 2 * this.random() - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const f = Math.sqrt((-2 * Math.log(s)) / s);
    this.spare = v * f;
    return mu + sd * u * f;
  }

  exponential(scale = 1) {
    // 1 - random() keeps the argument of log away from zero.
    return -scale * Math.log(1 - this.random());
  }

  int(a, b) {
    return a + Math.floor(this.random() * (b - a));
  }

  choice(items) {
    return items[Math.floor(this.random() * items.length)];
  }
}
