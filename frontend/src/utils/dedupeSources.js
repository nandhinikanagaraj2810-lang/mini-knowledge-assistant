const normalize = (text) => String(text ?? '').replace(/\s+/g, ' ').trim().toLowerCase();

/**
 * Return each unique source once, preserving order (first occurrence wins).
 *
 * A source is a duplicate if it has the same `chunk_id`, OR - as a fallback for
 * the same text stored under different chunk ids - the same title + excerpt.
 * Different chunks of the same document (different excerpt) are kept.
 */
export function dedupeSources(sources) {
  if (!Array.isArray(sources)) return [];
  const seenIds = new Set();
  const seenText = new Set();
  const unique = [];
  for (const src of sources) {
    if (!src) continue;
    const textKey = `${normalize(src.title)}\u0000${normalize(src.excerpt)}`;
    if ((src.chunk_id != null && seenIds.has(src.chunk_id)) || seenText.has(textKey)) continue;
    if (src.chunk_id != null) seenIds.add(src.chunk_id);
    seenText.add(textKey);
    unique.push(src);
  }
  return unique;
}
