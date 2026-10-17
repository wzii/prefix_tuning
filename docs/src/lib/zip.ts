/** Minimal ZIP (stored, no compression) for wrapping one file. Enough to round-trip our own archives. */
const TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(b: Uint8Array) {
  let c = 0xffffffff;
  for (let i = 0; i < b.length; i++) c = TABLE[(c ^ b[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

export function zipOne(name: string, data: Uint8Array): Blob {
  const fname = new TextEncoder().encode(name);
  const crc = crc32(data);
  const local = new DataView(new ArrayBuffer(30));
  local.setUint32(0, 0x04034b50, true);
  local.setUint16(4, 20, true);
  local.setUint16(8, 0, true); // stored
  local.setUint32(14, crc, true);
  local.setUint32(18, data.length, true);
  local.setUint32(22, data.length, true);
  local.setUint16(26, fname.length, true);
  const central = new DataView(new ArrayBuffer(46));
  central.setUint32(0, 0x02014b50, true);
  central.setUint16(4, 20, true);
  central.setUint16(6, 20, true);
  central.setUint32(16, crc, true);
  central.setUint32(20, data.length, true);
  central.setUint32(24, data.length, true);
  central.setUint16(28, fname.length, true);
  central.setUint32(42, 0, true); // local header offset
  const cdOffset = 30 + fname.length + data.length;
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, 1, true);
  end.setUint16(10, 1, true);
  end.setUint32(12, 46 + fname.length, true);
  end.setUint32(16, cdOffset, true);
  return new Blob([local.buffer, fname, data as BlobPart, central.buffer, fname, end.buffer], { type: 'application/zip' });
}

export const isZip = (buf: ArrayBuffer) => buf.byteLength > 4 && new DataView(buf).getUint32(0, true) === 0x04034b50;

/** First stored entry whose name ends with `ext`, read via the central directory. */
export function unzipFirst(buf: ArrayBuffer, ext: string): { name: string; data: ArrayBuffer } {
  const v = new DataView(buf);
  let e = buf.byteLength - 22;
  while (e >= 0 && v.getUint32(e, true) !== 0x06054b50) e--;
  if (e < 0) throw new Error('not a zip file');
  let p = v.getUint32(e + 16, true);
  const count = v.getUint16(e + 10, true);
  for (let i = 0; i < count; i++) {
    const method = v.getUint16(p + 10, true);
    const size = v.getUint32(p + 20, true);
    const nlen = v.getUint16(p + 28, true);
    const xlen = v.getUint16(p + 30, true);
    const clen = v.getUint16(p + 32, true);
    const off = v.getUint32(p + 42, true);
    const name = new TextDecoder().decode(new Uint8Array(buf, p + 46, nlen));
    if (name.endsWith(ext)) {
      if (method !== 0) throw new Error('compressed zip entries are not supported; unzip the file first');
      const start = off + 30 + v.getUint16(off + 26, true) + v.getUint16(off + 28, true);
      return { name, data: buf.slice(start, start + size) };
    }
    p += 46 + nlen + xlen + clen;
  }
  throw new Error(`no ${ext} file inside the zip`);
}
