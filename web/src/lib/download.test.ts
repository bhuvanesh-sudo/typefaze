// @vitest-environment node
import { describe, expect, it } from "vitest";
import { fetchWithProgress } from "./download";

function streamResponse(chunks: number[][], headers: Record<string, string> = {}, status = 200) {
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      for (const ch of chunks) c.enqueue(new Uint8Array(ch));
      c.close();
    },
  });
  return new Response(body, { status, headers });
}

describe("fetchWithProgress", () => {
  it("reports growing progress and returns all the bytes", async () => {
    const seen: [number, number | null][] = [];
    const bytes = await fetchWithProgress(
      "m.onnx",
      (loaded, total) => seen.push([loaded, total]),
      async () => streamResponse([[1, 2, 3], [4, 5], [6]], { "content-length": "6" }),
    );
    expect(Array.from(bytes)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(seen.map((s) => s[0])).toEqual([3, 5, 6]);
    expect(seen.every((s) => s[1] === 6)).toBe(true);
  });

  it("reports an unknown total as null", async () => {
    const seen: (number | null)[] = [];
    await fetchWithProgress("m.onnx", (_, t) => seen.push(t), async () => streamResponse([[1], [2]]));
    expect(seen).toEqual([null, null]);
  });

  it("fails clearly when the server says no", async () => {
    await expect(fetchWithProgress("m.onnx", () => {}, async () => streamResponse([], {}, 404))).rejects.toThrow(
      "Could not download m.onnx (HTTP 404)",
    );
  });
});
