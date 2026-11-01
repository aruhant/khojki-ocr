"use strict";

const elements = Object.fromEntries(Array.from(document.querySelectorAll("[id]"), (element) => [element.id.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase()), element]));
const terminalStatuses = new Set(["review", "blank", "done", "complete", "completed", "finished", "cancelled", "canceled", "failed", "error"]);
const state = {
  job: null,
  filename: "",
  pages: new Map(),
  selected: null,
  status: "",
  source: null,
  pollTimer: null,
  pollAttempts: 0,
  pollFailures: 0,
  polling: false,
  refreshPromise: null,
  requestControllers: new Set(),
  uploadController: null,
  generation: 0,
  zoom: 1,
  readOnly: false,
  dragDepth: 0,
  editing: false,
  storageWarning: false,
};

function showAlert(message = "") {
  elements.alert.textContent = message;
  elements.alert.hidden = !message;
}

function isTerminal(status = state.status) {
  return terminalStatuses.has(String(status || "").toLowerCase());
}

function label(value) {
  return String(value || "Waiting").replace(/[_-]/g, " ").replace(/^./, (letter) => letter.toUpperCase());
}

function jobUrl(suffix = "", job = state.job) {
  return `/api/jobs/${encodeURIComponent(job)}${suffix}`;
}

function serverMessage(payload, fallback) {
  if (typeof payload?.error?.message === "string") return payload.error.message;
  if (typeof payload?.error === "string") return payload.error;
  if (typeof payload?.detail === "string") return payload.detail;
  return fallback;
}

async function request(url, options = {}) {
  const controller = new AbortController();
  state.requestControllers.add(controller);
  const timeout = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    if (!response.ok) {
      let message = `Request failed (${response.status})`;
      try {
        message = serverMessage(await response.json(), message);
      } catch {}
      throw new Error(message);
    }
    return response;
  } finally {
    clearTimeout(timeout);
    state.requestControllers.delete(controller);
  }
}

function stopUpdates() {
  if (state.source) state.source.close();
  state.source = null;
  clearInterval(state.pollTimer);
  state.pollTimer = null;
  for (const controller of state.requestControllers) controller.abort();
  state.requestControllers.clear();
  state.polling = false;
  state.refreshPromise = null;
}

function pageText(page) {
  return page.edited ? page.editText : page.text;
}

function pageRecord(number) {
  if (!state.pages.has(number)) state.pages.set(number, { number, text: "", status: "waiting", edited: false, editText: "", review_required: false });
  return state.pages.get(number);
}

function restoreEdits() {
  try {
    const stored = JSON.parse(localStorage.getItem(`khojki-edits:${state.job}`) || "null");
    if (!stored || typeof stored !== "object") return;
    for (const [number, text] of Object.entries(stored)) {
      if (!Number.isInteger(Number(number)) || Number(number) < 1 || typeof text !== "string") continue;
      Object.assign(pageRecord(Number(number)), { edited: true, editText: text });
    }
  } catch {}
}

function persistEdits() {
  const edits = Object.fromEntries(Array.from(state.pages.values()).filter((page) => page.edited).map((page) => [page.number, page.editText]));
  try {
    localStorage.setItem(`khojki-edits:${state.job}`, JSON.stringify(edits));
    state.storageWarning = false;
  } catch {
    state.storageWarning = true;
  }
}

function updatePage(payload) {
  if (!payload || typeof payload !== "object") return;
  if (payload.page && typeof payload.page === "object") {
    updatePage(payload.page);
    return;
  }
  const number = Number(payload.number ?? payload.page);
  if (!Number.isInteger(number) || number < 1) return;
  const record = pageRecord(number);
  const text = payload.text ?? payload.raw_hypothesis ?? payload.draft_text ??
    (Array.isArray(payload.lines) ? payload.lines.map((line) => typeof line === "string" ? line : line?.raw_hypothesis ?? line?.text ?? "").join("\n") : undefined);
  if (typeof text === "string") {
    record.text = payload.stage === "recognition" ? record.text + (record.text ? "\n" : "") + text : text;
  }
  if (payload.status) record.status = String(payload.status).toLowerCase();
  if (payload.preview_url) record.preview_url = payload.preview_url;
  if (payload.review_required !== undefined) record.review_required = Boolean(payload.review_required);
  if (payload.error) record.error = String(payload.error);
  if (state.selected === null) state.selected = number;
}

function setProgress(payload) {
  const total = Number(payload.total ?? payload.total_pages ?? state.pages.size);
  const current = Number(payload.page && typeof payload.page === "object" ? payload.page.number ?? payload.page.page : payload.page ?? payload.number);
  if (Number.isInteger(total) && total > 0 && total <= 10000) {
    for (let number = 1; number <= total; number += 1) pageRecord(number);
    if (state.selected === null) state.selected = 1;
  }
  const parts = [];
  if (payload.stage) parts.push(label(payload.stage));
  if (Number.isFinite(current) && current > 0) parts.push(`Page ${current}${total > 0 ? ` of ${total}` : ""}`);
  if (payload.status && !parts.length) parts.push(label(payload.status));
  if (parts.length) elements.jobStatus.textContent = parts.join(" · ");
  const rawPercent = Number(payload.progress ?? payload.percent);
  const knownPercent = payload.progress !== undefined || payload.percent !== undefined;
  const completed = Array.from(state.pages.values()).filter((page) => isTerminal(page.status) || ["blank", "skipped", "ready", "review_required"].includes(page.status)).length;
  const precedingPages = Number.isInteger(current) && current > 0 ? current - 1 : 0;
  const percent = knownPercent && Number.isFinite(rawPercent) ? Math.max(0, Math.min(100, rawPercent)) : total > 0 ? Math.min(100, Math.floor(Math.max(completed, precedingPages) / total * 100)) : null;
  if (percent === null) {
    elements.progressTrack.removeAttribute("aria-valuenow");
  } else {
    elements.progressTrack.setAttribute("aria-valuenow", String(percent));
    elements.progressFill.style.width = `${percent}%`;
  }
}

function sortedPages() {
  return Array.from(state.pages.values()).sort((first, second) => first.number - second.number);
}

function updateActions() {
  const page = state.pages.get(state.selected);
  const hasText = Array.from(state.pages.values()).some((item) => pageText(item).trim());
  elements.download.disabled = !hasText || !isTerminal() || Boolean(state.uploadController);
  elements.copy.disabled = !page || !pageText(page).trim();
  elements.cancel.hidden = !state.job || isTerminal();
  elements.cancel.disabled = Boolean(state.uploadController);
  const pages = sortedPages();
  const index = pages.findIndex((item) => item.number === state.selected);
  elements.previous.disabled = index <= 0;
  elements.next.disabled = index < 0 || index >= pages.length - 1;
}

function renderPages() {
  const fragment = document.createDocumentFragment();
  for (const page of sortedPages()) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "page-button";
    if (page.error || ["failed", "error"].includes(page.status)) button.classList.add("failed");
    else if (page.review_required) button.classList.add("review");
    else if (page.text.trim()) button.classList.add("ready");
    button.textContent = String(page.number);
    button.setAttribute("aria-label", `Page ${page.number}, ${page.review_required ? "review required" : label(page.status)}`);
    if (page.number === state.selected) button.setAttribute("aria-current", "page");
    button.addEventListener("click", () => selectPage(page.number));
    fragment.append(button);
  }
  const scrollPosition = elements.pageList.scrollLeft;
  elements.pageList.replaceChildren(fragment);
  elements.pageList.scrollLeft = scrollPosition;
}

function safePreviewUrl(value, fallback) {
  if (!value) return fallback;
  try {
    const url = new URL(value, location.href);
    return url.origin === location.origin && ["http:", "https:"].includes(url.protocol) ? url.href : fallback;
  } catch {
    return fallback;
  }
}

function renderImage() {
  const page = state.pages.get(state.selected);
  if (!page) return;
  const url = safePreviewUrl(page.preview_url, jobUrl(`/pages/${page.number}.png`));
  const key = `${state.job}:${page.number}:${url}`;
  if (elements.pageImage.dataset.key === key) return;
  elements.pageImage.dataset.key = key;
  state.zoom = 1;
  elements.pageImage.classList.remove("zoomed");
  elements.pageImage.style.width = "";
  elements.pageImage.hidden = true;
  elements.imageMessage.hidden = false;
  elements.imageMessage.classList.remove("error");
  elements.imageMessage.textContent = "Loading page image";
  elements.pageImage.alt = `${state.filename}, page ${page.number}`;
  elements.pageImage.onload = () => {
    if (elements.pageImage.dataset.key !== key) return;
    elements.pageImage.hidden = false;
    elements.imageMessage.hidden = true;
    elements.zoomIn.disabled = false;
    elements.zoomOut.disabled = false;
    elements.fit.disabled = false;
  };
  elements.pageImage.onerror = () => {
    if (elements.pageImage.dataset.key !== key) return;
    elements.pageImage.hidden = true;
    elements.imageMessage.hidden = false;
    elements.imageMessage.classList.add("error");
    elements.imageMessage.textContent = isTerminal() ? "Page image unavailable" : "Waiting for page image";
    elements.zoomIn.disabled = true;
    elements.zoomOut.disabled = true;
    elements.fit.disabled = true;
    delete elements.pageImage.dataset.key;
  };
  elements.pageImage.src = url;
}

function renderTranscript(force = false) {
  const page = state.pages.get(state.selected);
  if (!page) return;
  const text = pageText(page);
  elements.pageNumber.textContent = String(page.number);
  if ((force || !state.editing || !page.edited) && elements.transcript.value !== text) elements.transcript.value = text;
  elements.reader.textContent = text;
  elements.transcript.disabled = false;
  elements.transcript.hidden = state.readOnly;
  elements.reader.hidden = !state.readOnly;
  elements.draftLabel.textContent = page.edited ? "Edited draft" : "Model draft";
  elements.saveStatus.textContent = page.edited ? state.storageWarning ? "Edits in this session only" : "Edits saved in this browser" : "Model draft";
  elements.characterCount.textContent = `${Array.from(text).length} characters`;
  const blank = !text.trim() && (["blank", "skipped", "done", "completed", "complete", "ready"].includes(page.status) || ["done", "complete", "completed", "finished"].includes(state.status));
  elements.pageStatus.textContent = page.error || (blank ? "No text detected" : label(page.status));
  elements.reviewBadge.hidden = !(page.review_required || page.error || page.text.trim());
  updateActions();
}

function render(force = false) {
  renderPages();
  renderImage();
  renderTranscript(force);
  updateActions();
}

function selectPage(number) {
  if (!state.pages.has(number)) return;
  state.selected = number;
  state.editing = false;
  render(true);
  elements.pageList.querySelector('[aria-current="page"]')?.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function applyFullState(payload) {
  if (typeof payload.filename === "string" && payload.filename && payload.filename !== state.filename) {
    state.filename = payload.filename;
    elements.filename.textContent = state.filename;
    rememberJob();
  }
  const pages = Array.isArray(payload.pages) ? payload.pages : [];
  for (const page of pages) updatePage(page);
  if (payload.status) state.status = String(payload.status).toLowerCase();
  setProgress({ ...payload, total: payload.total ?? payload.total_pages ?? pages.length });
  if (payload.error) showAlert(String(payload.error));
  render();
  if (isTerminal()) {
    stopUpdates();
    elements.jobStatus.textContent = label(state.status);
    if (["review", "blank", "done", "complete", "completed", "finished"].includes(state.status)) {
      elements.progressFill.style.width = "100%";
      elements.progressTrack.setAttribute("aria-valuenow", "100");
    }
    elements.progressTrack.classList.toggle("failed", ["failed", "error"].includes(state.status));
  }
}

async function refresh(generation = state.generation) {
  if (generation !== state.generation) return;
  if (state.refreshPromise?.generation === generation) return state.refreshPromise.promise;
  const pending = { generation, promise: null };
  pending.promise = (async () => {
    const response = await request(jobUrl());
    const payload = await response.json();
    if (generation === state.generation) applyFullState(payload);
  })();
  state.refreshPromise = pending;
  try {
    return await pending.promise;
  } finally {
    if (state.refreshPromise === pending) state.refreshPromise = null;
  }
}

function startPolling(generation) {
  if (state.pollTimer || isTerminal()) return;
  if (state.source) state.source.close();
  state.source = null;
  elements.jobStatus.textContent = "Live connection interrupted · Reconnecting";
  state.pollAttempts = 0;
  state.pollFailures = 0;
  state.pollTimer = setInterval(async () => {
    if (state.polling || generation !== state.generation) return;
    state.polling = true;
    state.pollAttempts += 1;
    try {
      await refresh(generation);
      if (generation === state.generation) state.pollFailures = 0;
    } catch (error) {
      if (generation !== state.generation) return;
      state.pollFailures += 1;
      if (state.pollFailures >= 120) {
        stopUpdates();
        showAlert("Connection unavailable. Reload to reconnect to this document.");
      } else {
        elements.jobStatus.textContent = "Connection interrupted · Retrying";
      }
    } finally {
      if (generation === state.generation) state.polling = false;
    }
    if (state.pollAttempts >= 2400 && !isTerminal() && generation === state.generation) {
      stopUpdates();
      showAlert("Live updates paused. Reload to reconnect to this document.");
    }
  }, 3000);
}

function startEvents(generation) {
  if (typeof EventSource === "undefined") {
    startPolling(generation);
    return;
  }
  const source = new EventSource(jobUrl("/events"));
  state.source = source;
  const handleEvent = async (event) => {
    if (generation !== state.generation) return;
    let payload;
    try { payload = JSON.parse(event.data); } catch { return; }
    if (!payload || typeof payload !== "object") return;
    if (payload.snapshot) {
      applyFullState(payload);
      return;
    }
    const status = String(payload.job_status ?? payload.status ?? "").toLowerCase();
    if (status && !isTerminal(status) && !["blank", "skipped", "ready", "review_required"].includes(status)) state.status = status;
    updatePage(payload);
    setProgress(payload);
    if (payload.error) showAlert(String(payload.error));
    render();
    if (isTerminal(status) || ["complete", "done"].includes(event.type)) {
      try { await refresh(generation); } catch {
        if (generation === state.generation) {
          startPolling(generation);
          showAlert("Final document state unavailable. Reload to reconnect before downloading.");
        }
      }
    }
  };
  source.addEventListener("progress", handleEvent);
  source.addEventListener("complete", handleEvent);
  source.addEventListener("done", handleEvent);
  source.onmessage = handleEvent;
  source.onerror = () => {
    if (generation === state.generation && !isTerminal()) startPolling(generation);
  };
}

function rememberJob() {
  const url = new URL(location.href);
  url.searchParams.set("job", state.job);
  history.replaceState(null, "", url);
  try { localStorage.setItem(`khojki-name:${state.job}`, state.filename); } catch {}
}

function activateJob(job, filename) {
  stopUpdates();
  state.generation += 1;
  state.job = String(job);
  state.filename = filename;
  state.pages.clear();
  state.selected = null;
  state.status = "queued";
  state.editing = false;
  state.storageWarning = false;
  elements.empty.hidden = true;
  elements.document.hidden = false;
  elements.filename.textContent = filename;
  elements.jobStatus.textContent = "Queued";
  elements.progressFill.style.width = "0%";
  elements.progressTrack.removeAttribute("aria-valuenow");
  elements.progressTrack.classList.remove("failed");
  elements.pageImage.hidden = true;
  elements.pageImage.onload = null;
  elements.pageImage.onerror = null;
  elements.pageImage.removeAttribute("src");
  delete elements.pageImage.dataset.key;
  elements.imageMessage.hidden = false;
  elements.imageMessage.textContent = "Waiting for page image";
  elements.imageMessage.classList.remove("error");
  elements.transcript.value = "";
  elements.reader.textContent = "";
  elements.transcript.disabled = true;
  elements.zoomIn.disabled = true;
  elements.zoomOut.disabled = true;
  elements.fit.disabled = true;
  elements.pageList.replaceChildren();
  elements.pageStatus.textContent = "Waiting";
  elements.pageNumber.textContent = "1";
  elements.characterCount.textContent = "0 characters";
  elements.draftLabel.textContent = "Model draft";
  elements.saveStatus.textContent = "Model draft";
  elements.reviewBadge.hidden = true;
  restoreEdits();
  rememberJob();
  updateActions();
  const generation = state.generation;
  startEvents(generation);
  refresh(generation).catch(() => {
    if (generation === state.generation && !isTerminal()) startPolling(generation);
  });
}

async function uploadFile(file) {
  if (!file) return;
  const validExtension = /\.(pdf|png|jpe?g)$/i.test(file.name);
  const validType = ["application/pdf", "image/png", "image/jpeg"].includes(file.type);
  if ((!validExtension && !validType) || !file.size) {
    showAlert("Choose a nonempty PDF, PNG, or JPEG file.");
    return;
  }
  if (state.uploadController) state.uploadController.abort();
  const controller = new AbortController();
  state.uploadController = controller;
  const previousJob = state.job;
  const previousStatus = state.status;
  const previousStatusText = elements.jobStatus.textContent;
  showAlert();
  elements.upload.disabled = true;
  elements.emptyUpload.disabled = true;
  elements.upload.querySelector("span:last-child").textContent = "Uploading";
  elements.emptyUpload.querySelector("span:last-child").textContent = "Uploading";
  elements.upload.setAttribute("aria-busy", "true");
  elements.jobStatus.textContent = "Uploading";
  updateActions();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const filenameHeader = /[^\u0020-\u007e]/.test(file.name) ? encodeURIComponent(file.name) : file.name;
    const response = await fetch("/api/jobs", { method: "POST", headers: { "Content-Type": "application/octet-stream", "X-Filename": filenameHeader }, body: file, signal: controller.signal });
    if (!response.ok) {
      let message = `Upload failed (${response.status})`;
      try { message = serverMessage(await response.json(), message); } catch {}
      throw new Error(String(message));
    }
    const payload = await response.json();
    const job = payload.id ?? payload.job_id;
    if ((typeof job !== "string" && typeof job !== "number") || !String(job)) throw new Error("The server did not return a document ID.");
    if (state.uploadController !== controller) return;
    activateJob(job, file.name);
    if (previousJob && !isTerminal(previousStatus) && String(previousJob) !== String(job)) {
      request(jobUrl("/cancel", previousJob), { method: "POST" }).catch(() => {
        showAlert("The previous document could not be cancelled and may still be processing.");
      });
    }
  } catch (error) {
    if (state.uploadController !== controller) return;
    elements.jobStatus.textContent = previousStatusText;
    showAlert(error.name === "AbortError" ? "Upload interrupted or timed out. Your previous document is unchanged." : error.message);
  } finally {
    clearTimeout(timeout);
    if (state.uploadController === controller) {
      state.uploadController = null;
      elements.upload.disabled = false;
      elements.emptyUpload.disabled = false;
      elements.upload.querySelector("span:last-child").textContent = "Upload document";
      elements.emptyUpload.querySelector("span:last-child").textContent = "Upload document";
      elements.upload.removeAttribute("aria-busy");
      updateActions();
    }
    elements.fileInput.value = "";
  }
}

async function cancelJob() {
  if (!state.job || isTerminal()) return;
  const generation = state.generation;
  elements.cancel.disabled = true;
  try {
    await request(jobUrl("/cancel"), { method: "POST" });
    if (generation !== state.generation) return;
    elements.jobStatus.textContent = "Cancellation requested";
    await refresh(generation);
  } catch (error) {
    if (generation === state.generation) showAlert(`Cancellation could not be confirmed: ${error.message}`);
  } finally {
    if (generation === state.generation) elements.cancel.disabled = false;
  }
}

async function downloadTranscript() {
  if (elements.download.disabled) return;
  const generation = state.generation;
  const filename = state.filename.replace(/\.[^.]+$/, "") || "transcript";
  const pages = sortedPages();
  elements.download.disabled = true;
  try {
    const edited = pages.some((page) => page.edited);
    const blob = edited ? new Blob([pages.map(pageText).join("\n\n")], { type: "text/plain;charset=utf-8" }) : await (await request(jobUrl("/transcript.txt"))).blob();
    if (generation !== state.generation) return;
    if (!blob.size) throw new Error("No transcript is available for this document.");
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${filename}${edited ? "-edited" : "-draft"}.txt`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    if (generation === state.generation) showAlert(`Download failed: ${error.message}`);
  } finally {
    if (generation === state.generation) updateActions();
  }
}

function setMode(readOnly) {
  state.readOnly = readOnly;
  elements.editMode.setAttribute("aria-pressed", String(!readOnly));
  elements.readMode.setAttribute("aria-pressed", String(readOnly));
  renderTranscript();
}

function setView(view) {
  elements.document.dataset.view = view;
  elements.pageTab.setAttribute("aria-selected", String(view === "page"));
  elements.textTab.setAttribute("aria-selected", String(view === "text"));
  elements.pageTab.tabIndex = view === "page" ? 0 : -1;
  elements.textTab.tabIndex = view === "text" ? 0 : -1;
}

function zoomImage(amount) {
  if (elements.pageImage.hidden) return;
  if (state.zoom === 1) state.fitWidth = elements.pageImage.getBoundingClientRect().width;
  state.zoom = Math.max(1, Math.min(4, state.zoom + amount));
  elements.pageImage.classList.toggle("zoomed", state.zoom > 1);
  elements.pageImage.style.width = state.zoom > 1 ? `${state.fitWidth * state.zoom}px` : "";
}

for (const button of [elements.upload, elements.emptyUpload]) button.addEventListener("click", () => elements.fileInput.click());
elements.upload.setAttribute("aria-label", "Upload document");
elements.upload.title = "Upload document";
elements.fileInput.addEventListener("change", () => uploadFile(elements.fileInput.files[0]));
elements.cancel.addEventListener("click", cancelJob);
elements.download.addEventListener("click", downloadTranscript);
elements.previous.addEventListener("click", () => { const pages = sortedPages(); const index = pages.findIndex((page) => page.number === state.selected); if (index > 0) selectPage(pages[index - 1].number); });
elements.next.addEventListener("click", () => { const pages = sortedPages(); const index = pages.findIndex((page) => page.number === state.selected); if (index < pages.length - 1) selectPage(pages[index + 1].number); });
elements.editMode.addEventListener("click", () => setMode(false));
elements.readMode.addEventListener("click", () => setMode(true));
elements.pageTab.addEventListener("click", () => setView("page"));
elements.textTab.addEventListener("click", () => setView("text"));
for (const tab of [elements.pageTab, elements.textTab]) tab.addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  event.preventDefault();
  const view = event.key === "Home" ? "page" : event.key === "End" ? "text" : elements.document.dataset.view === "page" ? "text" : "page";
  setView(view);
  (view === "page" ? elements.pageTab : elements.textTab).focus();
});
elements.transcript.addEventListener("focus", () => { state.editing = true; });
elements.transcript.addEventListener("blur", () => { state.editing = false; renderTranscript(); });
elements.transcript.addEventListener("input", () => {
  const page = state.pages.get(state.selected);
  if (!page) return;
  page.edited = true;
  page.editText = elements.transcript.value;
  persistEdits();
  renderTranscript();
});
elements.copy.addEventListener("click", async () => {
  const page = state.pages.get(state.selected);
  if (!page) return;
  try {
    if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
    await navigator.clipboard.writeText(pageText(page));
    elements.saveStatus.textContent = "Page text copied";
  } catch {
    setMode(false);
    elements.transcript.focus();
    elements.transcript.select();
    elements.saveStatus.textContent = "Clipboard unavailable · Text selected";
  }
});
elements.zoomIn.addEventListener("click", () => zoomImage(.5));
elements.zoomOut.addEventListener("click", () => zoomImage(-.5));
elements.fit.addEventListener("click", () => {
  state.zoom = 1;
  elements.pageImage.classList.remove("zoomed");
  elements.pageImage.style.width = "";
  elements.imageViewport.scrollTo(0, 0);
});
window.addEventListener("dragenter", (event) => {
  if (!Array.from(event.dataTransfer?.types || []).includes("Files")) return;
  event.preventDefault();
  state.dragDepth += 1;
  elements.dropOverlay.hidden = false;
});
window.addEventListener("dragover", (event) => {
  if (Array.from(event.dataTransfer?.types || []).includes("Files")) { event.preventDefault(); event.dataTransfer.dropEffect = "copy"; }
});
window.addEventListener("dragleave", () => { state.dragDepth = Math.max(0, state.dragDepth - 1); if (!state.dragDepth) elements.dropOverlay.hidden = true; });
window.addEventListener("drop", (event) => {
  if (!event.dataTransfer?.files.length) return;
  event.preventDefault();
  state.dragDepth = 0;
  elements.dropOverlay.hidden = true;
  if (event.dataTransfer.files.length !== 1) { showAlert("Choose one document at a time."); return; }
  uploadFile(event.dataTransfer.files[0]);
});
window.addEventListener("pagehide", () => { stopUpdates(); state.uploadController?.abort(); });
window.addEventListener("pageshow", (event) => { if (event.persisted && state.job && !isTerminal()) { startEvents(state.generation); refresh().catch(() => startPolling(state.generation)); } });

setView("page");
if (window.lucide?.createIcons) window.lucide.createIcons();
const resumedJob = new URL(location.href).searchParams.get("job");
if (resumedJob) {
  let filename = "Document";
  try { filename = localStorage.getItem(`khojki-name:${resumedJob}`) || filename; } catch {}
  activateJob(resumedJob, filename);
}
fetch("/api/config").then((response) => response.ok ? response.json() : null).then((config) => {
  if (!config) return;
  if (typeof config.model_name === "string") elements.draftLabel.title = config.model_name;
  if (config.available === false || config.model_available === false) showAlert(config.error || "Recognition model unavailable on this server.");
}).catch(() => {});