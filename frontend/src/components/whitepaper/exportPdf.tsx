import { toCanvas } from "html-to-image";
import { pdf, Document, Page, Image as PDFImage } from "@react-pdf/renderer";

// ─────────────────────────────────────────────────────────────────────────────
// Shared whitepaper → PDF exporter.
//
// Strategy: capture each top-level section as ONE canvas (preserving its
// internal text rendering — h3 subheadings, FactTable rows, Callouts all
// land on the canvas exactly as the browser draws them), then bin-pack
// the section canvases into pages. A section taller than one page gets
// sliced at element boundaries; the slices are also bin-packable, so a
// tall section's tail can share a page with the next short section.
//
// Why per-section instead of per-block: capturing individual <h3>s in
// isolation breaks their text rendering inside html-to-image's
// foreignObject context. Whole-section capture sidesteps this entirely
// because the h3 stays nested in the section's DOM during the clone.
// ─────────────────────────────────────────────────────────────────────────────

const PDF_PIXEL_RATIO = 2;
const PDF_PAGE_STYLE = { padding: "10mm" } as const;
const PDF_PAGE_STYLE_CENTERED = {
  padding: "10mm",
  flexDirection: "row" as const,
  alignItems: "center" as const,
  justifyContent: "center" as const,
};
const PDF_IMAGE_FILL_W = { width: "100%" } as const;
const PDF_IMAGE_FILL_H = { height: "100%" } as const;
const A4_CONTENT_ASPECT = 190 / 277;

/** Vertical canvas pixels between adjacent mini-blocks on a packed
 *  page. Mini-blocks already include their original surrounding
 *  margins (slices are sub-rectangles of the section's canvas, which
 *  was captured with all padding + margins intact), so an extra gap
 *  would over-space them. Set to 0 — the natural margins do the work. */
const INTER_BLOCK_GAP_PX = 0;

/** Selectors whose bottom edges are safe break points inside a
 *  section. ``.card-title`` is the section heading, ``.callout`` /
 *  ``figure`` / ``table`` are atomic blocks, ``hr`` is an explicit
 *  operator break, and ``p``/``h3``/``tr``/``li`` give finer
 *  granularity for paragraph-and-heading-heavy sections. */
const INTRA_SECTION_BREAK_SELECTORS = [
  ".card-title",
  "figure",
  "hr",
  ".callout",
  "table",
  "tr",
  "li",
  "p",
  "h3",
] as const;

interface ExportWhitepaperPdfOptions {
  /** The element whose top-level children get captured. */
  container: HTMLElement;
  /** Output filename (with .pdf extension). */
  filename: string;
  /** Title metadata embedded in the generated PDF. */
  title: string;
  /** Optional element id whose ``style.display`` is set to ``none`` for
   *  the duration of the capture. Use this for the "Export PDF" button
   *  itself so its layout slot collapses identically in the live DOM
   *  and the captured canvas. */
  hideElementId?: string;
}

interface PageSlice {
  src: string;
  /** Aspect ratio = width / height. Decides whether the slice fits
   *  the page by width (default) or needs to be height-bound (slice
   *  taller than the A4 content area). */
  aspectRatio: number;
}

// ─────────────────────────────────────────────────────────────────────
// Slicing helpers (used only when a single section exceeds page height)
// ─────────────────────────────────────────────────────────────────────

/** Bottom-edge offsets (in canvas pixels) of every safe-break element
 *  inside ``root``. Returned list is deduplicated and sorted ascending. */
function collectSafeBreakPoints(
  root: HTMLElement, pixelRatio: number,
): number[] {
  const rootRect = root.getBoundingClientRect();
  const rootHeightPx = Math.round(rootRect.height * pixelRatio);
  const set = new Set<number>([0, rootHeightPx]);
  const els = root.querySelectorAll(INTRA_SECTION_BREAK_SELECTORS.join(","));
  els.forEach((el) => {
    const rect = (el as HTMLElement).getBoundingClientRect();
    const bottom = (rect.bottom - rootRect.top) * pixelRatio;
    if (bottom > 0 && bottom <= rootHeightPx) {
      set.add(Math.round(bottom));
    }
  });
  return Array.from(set).sort((a, b) => a - b);
}

/** Cut a parent canvas into sub-canvases at the planned boundaries. */
function sliceCanvas(
  canvas: HTMLCanvasElement,
  pages: ReadonlyArray<{ start: number; end: number }>,
): HTMLCanvasElement[] {
  return pages.map(({ start, end }) => {
    const sliceCanvas = document.createElement("canvas");
    sliceCanvas.width = canvas.width;
    sliceCanvas.height = end - start;
    const ctx = sliceCanvas.getContext("2d")!;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, sliceCanvas.width, sliceCanvas.height);
    ctx.drawImage(canvas, 0, -start);
    return sliceCanvas;
  });
}

// ─────────────────────────────────────────────────────────────────────
// Capture + bin-pack
// ─────────────────────────────────────────────────────────────────────

/** Minimum slice height (canvas pixels) below which a slice gets
 *  merged with its neighbour. Tiny slices come from nested elements
 *  whose bottoms are within a few px of each other (e.g., a ``<p>``
 *  inside a ``.callout`` whose bottom is right above the callout's
 *  bottom). Below this threshold, slicing adds noise without helping
 *  packing. */
const MIN_SLICE_HEIGHT_PX = 8;

/** Capture every top-level child of ``container`` and slice it into
 *  mini-blocks at every internal element boundary. Mini-blocks are
 *  sub-rectangles of the section's full canvas — internal text
 *  rendering (including ``<h3>`` headings) is preserved exactly.
 *  This fine granularity is what enables the bin-packer to fill
 *  pages densely; whole-section blocks left ~50% trailing whitespace
 *  whenever a section was too tall for the remaining page space. */
async function captureBlocks(
  container: HTMLElement,
  pixelRatio: number,
  // pageHeightPx kept for symmetry; not used now that slicing is
  // uniform across all sections rather than just tall ones.
  _pageHeightPx: number,
): Promise<HTMLCanvasElement[]> {
  const sections: HTMLElement[] = [];
  for (const child of Array.from(container.children)) {
    if (!(child instanceof HTMLElement)) continue;
    if (child.getBoundingClientRect().height === 0) continue;
    sections.push(child);
  }

  const sectionCanvases = await Promise.all(
    sections.map((s) => toCanvas(s, {
      pixelRatio,
      backgroundColor: "#ffffff",
    })),
  );

  const blocks: HTMLCanvasElement[] = [];
  for (let i = 0; i < sections.length; i++) {
    const canvas = sectionCanvases[i];
    const breakpoints = collectSafeBreakPoints(sections[i], pixelRatio);
    if (breakpoints.length <= 2) {
      blocks.push(canvas);
      continue;
    }

    // Generate one slice per adjacent breakpoint pair, dropping
    // sub-threshold slices to avoid noise from nested elements with
    // near-identical bottoms.
    const pieces: Array<{ start: number; end: number }> = [];
    let prevEnd = breakpoints[0];
    for (let j = 1; j < breakpoints.length; j++) {
      const bp = breakpoints[j];
      if (bp - prevEnd < MIN_SLICE_HEIGHT_PX) continue;
      pieces.push({ start: prevEnd, end: bp });
      prevEnd = bp;
    }
    if (prevEnd < canvas.height) {
      pieces.push({ start: prevEnd, end: canvas.height });
    }
    blocks.push(...sliceCanvas(canvas, pieces));
  }
  return blocks;
}

/** Stack a list of block canvases onto a page canvas, centred
 *  horizontally with ``INTER_BLOCK_GAP_PX`` between adjacent blocks. */
function compositePage(
  blocks: HTMLCanvasElement[],
  pageWidth: number,
): PageSlice {
  const pageHeight = blocks.reduce(
    (acc, c, i) => acc + c.height + (i > 0 ? INTER_BLOCK_GAP_PX : 0),
    0,
  );
  const pageCanvas = document.createElement("canvas");
  pageCanvas.width = pageWidth;
  pageCanvas.height = pageHeight;
  const ctx = pageCanvas.getContext("2d")!;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, pageWidth, pageHeight);
  let y = 0;
  for (let i = 0; i < blocks.length; i++) {
    const c = blocks[i];
    const x = Math.max(0, Math.round((pageWidth - c.width) / 2));
    if (i > 0) y += INTER_BLOCK_GAP_PX;
    ctx.drawImage(c, x, y);
    y += c.height;
  }
  return {
    src: pageCanvas.toDataURL("image/png"),
    aspectRatio: pageCanvas.width / pageCanvas.height,
  };
}

/** Bin-pack blocks into page slices. Each page accumulates blocks
 *  until adding the next would exceed the A4 budget. Blocks taller
 *  than one page (rare) flush the current page and emit on their own
 *  page (scaled-to-fit on render). */
function packBlocksIntoPages(
  blocks: HTMLCanvasElement[],
  pageWidth: number,
  pageHeightPx: number,
): PageSlice[] {
  const slices: PageSlice[] = [];
  let pageBlocks: HTMLCanvasElement[] = [];
  let pageHeight = 0;

  const flush = () => {
    if (pageBlocks.length === 0) return;
    slices.push(compositePage(pageBlocks, pageWidth));
    pageBlocks = [];
    pageHeight = 0;
  };

  for (const block of blocks) {
    if (block.height > pageHeightPx) {
      flush();
      slices.push({
        src: block.toDataURL("image/png"),
        aspectRatio: block.width / block.height,
      });
      continue;
    }
    const projected = pageBlocks.length === 0
      ? block.height
      : pageHeight + INTER_BLOCK_GAP_PX + block.height;
    if (projected > pageHeightPx) {
      flush();
      pageBlocks.push(block);
      pageHeight = block.height;
    } else {
      pageBlocks.push(block);
      pageHeight = projected;
    }
  }
  flush();
  return slices;
}

// ─────────────────────────────────────────────────────────────────────
// Public entry
// ─────────────────────────────────────────────────────────────────────

export async function exportWhitepaperPdf(
  opts: ExportWhitepaperPdfOptions,
): Promise<void> {
  const { container, filename, title, hideElementId } = opts;

  const hideEl = hideElementId ? document.getElementById(hideElementId) : null;
  const prevDisplay = hideEl?.style.display ?? "";
  if (hideEl) hideEl.style.display = "none";

  try {
    const topLevel = Array.from(container.children).filter((el): el is HTMLElement => {
      if (!(el instanceof HTMLElement)) return false;
      const rect = el.getBoundingClientRect();
      return rect.height > 0 && rect.width > 0;
    });
    if (topLevel.length === 0) return;
    const refWidth = topLevel[0].getBoundingClientRect().width;
    // A4 usable area: 190mm × 277mm (10mm margins each side).
    const pageHeightPx = Math.floor(refWidth * PDF_PIXEL_RATIO * (277 / 190));

    const blocks = await captureBlocks(container, PDF_PIXEL_RATIO, pageHeightPx);
    if (blocks.length === 0) return;
    const pageWidth = Math.max(...blocks.map((b) => b.width));
    const slices = packBlocksIntoPages(blocks, pageWidth, pageHeightPx);

    const blob = await pdf(
      <Document title={title}>
        {slices.map(({ src, aspectRatio }, i) => {
          const fillByWidth = aspectRatio >= A4_CONTENT_ASPECT;
          return (
            <Page
              key={i}
              size="A4"
              style={fillByWidth ? PDF_PAGE_STYLE : PDF_PAGE_STYLE_CENTERED}
            >
              <PDFImage
                src={src}
                style={fillByWidth ? PDF_IMAGE_FILL_W : PDF_IMAGE_FILL_H}
              />
            </Page>
          );
        })}
      </Document>,
    ).toBlob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 200);
  } finally {
    if (hideEl) hideEl.style.display = prevDisplay;
  }
}
