/** Download a file while reporting bytes received. `total` is null when the server doesn't say. */
export async function fetchWithProgress(
  url: string,
  onProgress: (loaded: number, total: number | null) => void,
  fetchImpl: (url: string) => Promise<Response> = (u) => fetch(u),
): Promise<Uint8Array> {
  const res = await fetchImpl(url);
  if (!res.ok) throw new Error(`Could not download ${url} (HTTP ${res.status})`);
  const len = Number(res.headers.get("content-length"));
  const total = Number.isFinite(len) && len > 0 ? len : null;
  if (!res.body) {
    const buf = new Uint8Array(await res.arrayBuffer());
    onProgress(buf.length, total ?? buf.length);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}
