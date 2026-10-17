/**
 * Minimal safetensors reader/writer (https://github.com/huggingface/safetensors):
 * 8-byte little-endian header length, a JSON header, then raw little-endian tensor bytes.
 * Only F32 is needed here.
 */
export interface Tensor {
  shape: number[];
  data: Float32Array;
}

export function encodeSafetensors(tensors: Record<string, Tensor>, metadata: Record<string, string>): Uint8Array {
  const header: Record<string, unknown> = { __metadata__: metadata };
  let offset = 0;
  const names = Object.keys(tensors);
  for (const name of names) {
    const bytes = tensors[name].data.length * 4;
    header[name] = { dtype: 'F32', shape: tensors[name].shape, data_offsets: [offset, offset + bytes] };
    offset += bytes;
  }
  let json = JSON.stringify(header);
  json += ' '.repeat((8 - (json.length % 8)) % 8); // keep the data 8-byte aligned
  const head = new TextEncoder().encode(json);
  const out = new Uint8Array(8 + head.length + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(head.length), true);
  out.set(head, 8);
  let pos = 8 + head.length;
  for (const name of names) {
    const d = tensors[name].data;
    out.set(new Uint8Array(d.buffer, d.byteOffset, d.byteLength), pos);
    pos += d.byteLength;
  }
  return out;
}

export function decodeSafetensors(buf: ArrayBuffer): { tensors: Record<string, Tensor>; metadata: Record<string, string> } {
  if (buf.byteLength < 8) throw new Error('not a safetensors file');
  const n = Number(new DataView(buf).getBigUint64(0, true));
  if (n <= 0 || 8 + n > buf.byteLength) throw new Error('not a safetensors file');
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  const base = 8 + n;
  const tensors: Record<string, Tensor> = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const { dtype, shape, data_offsets } = info as { dtype: string; shape: number[]; data_offsets: [number, number] };
    if (dtype !== 'F32') throw new Error(`unsupported dtype ${dtype}`);
    const [a, b] = data_offsets;
    tensors[name] = { shape, data: new Float32Array(buf.slice(base + a, base + b)) };
  }
  return { tensors, metadata: (header.__metadata__ ?? {}) as Record<string, string> };
}

export async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  try {
    const h = await crypto.subtle.digest('SHA-256', bytes);
    return Array.from(new Uint8Array(h), (x) => x.toString(16).padStart(2, '0')).join('');
  } catch {
    // no SubtleCrypto (insecure context): fall back to a 32-bit FNV hash, clearly shorter
    let h = 2166136261 >>> 0;
    const u = new Uint8Array(bytes);
    for (let i = 0; i < u.length; i++) h = Math.imul(h ^ u[i], 16777619) >>> 0;
    return h.toString(16).padStart(8, '0');
  }
}
