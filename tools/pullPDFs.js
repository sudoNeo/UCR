(async () => {
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  const ORIGIN = location.origin;
  const COURSE_ID =
    window.ENV?.COURSE_ID ||
    location.pathname.match(/\/courses\/(\d+)/)?.[1] ||
    "PUT_COURSE_ID_HERE";

  const PREFIX_WITH_MODULE = true;
  
  const MS_BETWEEN_DOWNLOADS = 2000;
  
  const BATCH_SIZE = 5;
  const MS_BETWEEN_BATCHES = 3000;

  if (!COURSE_ID || COURSE_ID === "PUT_COURSE_ID_HERE") {
    console.error("Could not determine COURSE_ID. Set COURSE_ID at the top and run again.");
    return;
  }

  const parseNextLink = (linkHeader) => {
    if (!linkHeader) return null;
    const parts = linkHeader.split(",").map(s => s.trim());
    for (const p of parts) {
      const m = p.match(/^<([^>]+)>\s*;\s*rel="([^"]+)"/);
      if (m && m[2] === "next") return m[1];
    }
    return null;
  };

  const fetchAllPages = async (url) => {
    const all = [];
    let next = url;

    while (next) {
      const res = await fetch(next, { credentials: "include" });
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`Fetch failed ${res.status} ${res.statusText} for ${next}\n${txt.slice(0, 300)}`);
      }
      const data = await res.json();
      if (Array.isArray(data)) all.push(...data);
      else all.push(data);
      next = parseNextLink(res.headers.get("Link"));
    }

    return all;
  };

  const safeName = (s) =>
    String(s || "")
      .replace(/[\/\\:*?"<>|]/g, "_")
      .replace(/\s+/g, " ")
      .trim();

  const downloadViaAnchor = (url, filename) => {
    return new Promise((resolve) => {
      const a = document.createElement("a");
      a.href = url;
      a.download = filename || "";
      a.rel = "noopener";
      a.style.display = 'none';
      document.body.appendChild(a);
      
      // Click and give it time to process
      a.click();
      
      setTimeout(() => {
        a.remove();
        resolve();
      }, 500);
    });
  };

  const getFileInfo = async (fileId) => {
    let res = await fetch(`${ORIGIN}/api/v1/files/${fileId}`, { credentials: "include" });

    if (!res.ok) {
      res = await fetch(`${ORIGIN}/api/v1/courses/${COURSE_ID}/files/${fileId}`, { credentials: "include" });
    }

    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(`File info fetch failed for fileId=${fileId}: ${res.status} ${res.statusText}\n${txt.slice(0, 300)}`);
    }

    return res.json();
  };

  console.log("Canvas PDF downloader starting...");
  console.log({ ORIGIN, COURSE_ID });

  // 1) Get all modules
  const modulesUrl = `${ORIGIN}/api/v1/courses/${COURSE_ID}/modules?per_page=100`;
  const modules = await fetchAllPages(modulesUrl);

  console.log(`Found ${modules.length} modules.`);

  // 2) For each module, get items
  const fileCandidates = [];
  for (const mod of modules) {
    const itemsUrl = `${ORIGIN}/api/v1/courses/${COURSE_ID}/modules/${mod.id}/items?per_page=100`;
    const items = await fetchAllPages(itemsUrl);

    for (const it of items) {
      if (it?.type === "File" && it?.content_id) {
        fileCandidates.push({
          moduleId: mod.id,
          moduleName: mod.name,
          itemTitle: it.title,
          fileId: it.content_id
        });
      }
    }
  }

  // Dedup by fileId
  const byId = new Map();
  for (const c of fileCandidates) {
    if (!byId.has(c.fileId)) byId.set(c.fileId, c);
  }
  const uniqueCandidates = [...byId.values()];

  console.log(`Found ${uniqueCandidates.length} unique file items in modules.`);

  // 3) Resolve to actual PDFs
  let pdfs = [];
  for (const c of uniqueCandidates) {
    try {
      const info = await getFileInfo(c.fileId);

      const contentType = info.content_type || info["content-type"] || "";
      const filename = info.filename || info.display_name || c.itemTitle || `file_${c.fileId}`;
      const isPdf =
        String(filename).toLowerCase().endsWith(".pdf") ||
        String(contentType).toLowerCase().includes("pdf");

      if (!isPdf) continue;

      const baseDownloadUrl = info.url || info.download_url;
      if (!baseDownloadUrl) {
        console.warn("No download URL for:", c.fileId, info);
        continue;
      }

      const modulePrefix = PREFIX_WITH_MODULE ? `${safeName(c.moduleName)} - ` : "";
      const outName = safeName(modulePrefix + filename);

      pdfs.push({
        fileId: c.fileId,
        module: c.moduleName,
        filename: outName,
        url: baseDownloadUrl
      });
    } catch (e) {
      console.warn("Skipping file due to error:", c.fileId, e);
    }
  }

  console.log(`Ready to download ${pdfs.length} PDFs.`);
  console.table(pdfs.map(p => ({ fileId: p.fileId, module: p.module, filename: p.filename })));

  // 4) Download in batches
  for (let i = 0; i < pdfs.length; i += BATCH_SIZE) {
    const batch = pdfs.slice(i, Math.min(i + BATCH_SIZE, pdfs.length));
    const batchNum = Math.floor(i / BATCH_SIZE) + 1;
    const totalBatches = Math.ceil(pdfs.length / BATCH_SIZE);
    
    console.log(`\n=== BATCH ${batchNum}/${totalBatches} ===`);
    
    for (let j = 0; j < batch.length; j++) {
      const p = batch[j];
      const overallIndex = i + j + 1;
      console.log(`[${overallIndex}/${pdfs.length}] Downloading: ${p.filename}`);
      
      try {
        await downloadViaAnchor(p.url, p.filename);
        await sleep(MS_BETWEEN_DOWNLOADS);
      } catch (e) {
        console.error(`Failed to download ${p.filename}:`, e);
      }
    }
    
    // Pause between batches (except for the last one)
    if (i + BATCH_SIZE < pdfs.length) {
      console.log(`Batch complete. Pausing ${MS_BETWEEN_BATCHES}ms before next batch...`);
      await sleep(MS_BETWEEN_BATCHES);
    }
  }

  console.log("\nDone! Check your downloads folder.");
  console.log("If some downloads failed, check the console for errors.");
})();