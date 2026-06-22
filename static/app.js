/**
 * app.js
 * ------
 * Drives the Audio Transcription Filler UI.
 * Talks to the Flask backend (app.py) via fetch() calls.
 */

const pageListEl = document.getElementById("page-list");
const toastEl = document.getElementById("toast");

// Tracks in-progress MediaRecorder sessions, keyed by page_key
const activeRecorders = {};

// ── Screen switching (landing <-> main dashboard) ────────────
function showScreen(screenId) {
    document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
    document.getElementById(screenId).classList.add("active");
}

// ── Toast notifications ─────────────────────────────────────
function showToast(message, type = "") {
    toastEl.textContent = message;
    toastEl.className = "toast show " + type;
    setTimeout(() => {
        toastEl.className = "toast";
    }, 3000);
}

// ── Fetch current status from backend and render ────────────
async function loadStatus() {
    try {
        const res = await fetch("/api/status");
        const data = await res.json();
        renderSummary(data.summary);
        renderPages(data.pages);
    } catch (err) {
        showToast("Failed to load status: " + err.message, "error");
    }
}

function renderSummary(summary) {
    document.getElementById("stat-total").textContent = summary.total;
    document.getElementById("stat-pending").textContent = summary.pending;
    document.getElementById("stat-transcribed").textContent = summary.transcribed;
    document.getElementById("stat-confirmed").textContent = summary.confirmed;
    document.getElementById("stat-skipped").textContent = summary.skipped;

    const percent = summary.total > 0
        ? Math.round((summary.confirmed / summary.total) * 100)
        : 0;

    document.getElementById("dash-progress-fill").style.width = `${percent}%`;
    document.getElementById("dash-progress-label").textContent = `${percent}% complete`;
}

function renderPages(pages) {
    pageListEl.innerHTML = "";

    pages.forEach((page) => {
        const card = document.createElement("div");
        card.className = "page-card" + (page.is_null ? "" : " is-ok");
        card.id = `card-${page.page_key}`;

        const badgeClass = {
            "ok": "badge-ok",
            "pending": "badge-pending",
            "transcribed": "badge-transcribed",
            "confirmed": "badge-confirmed",
            "skipped": "badge-skipped"
        }[page.status] || "badge-pending";

        const badgeLabel = {
            "ok": "OCR OK",
            "pending": "Needs audio",
            "transcribed": "Awaiting confirm",
            "confirmed": "Confirmed",
            "skipped": "Skipped"
        }[page.status] || page.status;

        let contentHtml;
        if (page.content) {
            contentHtml = `<p class="page-content">${escapeHtml(page.content)}</p>`;
        } else {
            contentHtml = `<p class="page-content placeholder">No text — OCR failed on this page.</p>`;
        }

        let actionHtml = "";
        if (page.is_null && page.status !== "confirmed") {
            actionHtml = buildUploadZone(page);
        }

        card.innerHTML = `
            <div class="page-card-head">
                <span class="page-key">${page.page_key}</span>
                <span class="page-badge ${badgeClass}">${badgeLabel}</span>
            </div>
            ${contentHtml}
            ${actionHtml}
        `;

        pageListEl.appendChild(card);

        if (page.is_null && page.status !== "confirmed") {
            wireUploadZone(page.page_key);
        }
        if (page.status === "transcribed") {
            showPreview(page.page_key, page.transcribed_text);
        }
    });
}

function buildUploadZone(page) {
    return `
        <div class="page-action">
            <div class="upload-row">
                <label class="btn btn-outline btn-small" for="audio-${page.page_key}">
                    Upload audio
                    <input type="file" id="audio-${page.page_key}" accept="audio/*" hidden>
                </label>
                <button class="btn btn-outline btn-small" id="record-${page.page_key}">🎙 Record</button>
                <span class="file-label" id="filename-${page.page_key}">No file selected</span>
                <button class="btn btn-outline btn-small" id="skip-${page.page_key}">Skip page</button>
            </div>
            <div class="progress-text" id="progress-${page.page_key}">Transcribing — this may take a while for long audio...</div>
            <div class="preview-box" id="preview-${page.page_key}">
                <div class="preview-label">Transcription preview — edit if needed</div>
                <textarea class="preview-text" id="preview-text-${page.page_key}"></textarea>
                <div class="preview-actions">
                    <button class="btn btn-outline btn-small" id="edit-${page.page_key}">✏️ Edit with keyboard</button>
                    <button class="btn btn-primary btn-small" id="confirm-${page.page_key}">Confirm &amp; fill</button>
                </div>
            </div>
        </div>
    `;
}

function wireUploadZone(pageKey) {
    const fileInput = document.getElementById(`audio-${pageKey}`);
    const fileLabel = document.getElementById(`filename-${pageKey}`);
    const skipBtn = document.getElementById(`skip-${pageKey}`);
    const recordBtn = document.getElementById(`record-${pageKey}`);

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            fileLabel.textContent = fileInput.files[0].name;
            uploadAndTranscribe(pageKey, fileInput.files[0]);
        }
    });

    skipBtn.addEventListener("click", () => skipPage(pageKey));
    recordBtn.addEventListener("click", () => toggleRecording(pageKey, recordBtn, fileLabel));
}

async function uploadAndTranscribe(pageKey, file) {
    const progressEl = document.getElementById(`progress-${pageKey}`);
    progressEl.classList.add("active");

    const formData = new FormData();
    formData.append("audio", file);

    try {
        const res = await fetch(`/api/transcribe/${pageKey}`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();

        progressEl.classList.remove("active");

        if (!res.ok) {
            showToast(data.error || "Transcription failed", "error");
            return;
        }

        showPreview(pageKey, data.transcribed_text);
        showToast(`Transcription ready for ${pageKey}`, "success");
        loadStatus(); // refresh summary counts

    } catch (err) {
        progressEl.classList.remove("active");
        showToast("Upload failed: " + err.message, "error");
    }
}

// ── Microphone recording ─────────────────────────────────────
async function toggleRecording(pageKey, recordBtn, fileLabel) {
    const active = activeRecorders[pageKey];

    if (active) {
        // Already recording -> stop it. The actual upload happens
        // in the recorder's onstop handler below.
        active.recorder.stop();
        return;
    }

    let stream;
    try {
        // Explicit constraints instead of bare `audio: true` —
        // this is what was causing poor recording quality.
        // The browser default leaves echo, noise, and gain completely
        // unprocessed, and lets the OS pick whatever sample rate it wants.
        stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                channelCount: 1,        // mono — matches what Whisper needs anyway
                sampleRate: 16000        // ask for 16kHz directly, avoids resampling loss later
            }
        });
    } catch (err) {
        showToast("Microphone access denied or unavailable", "error");
        return;
    }

    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";

    const recorder = new MediaRecorder(stream, { mimeType });
    const chunks = [];
    const startedAt = Date.now();

    recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        clearInterval(activeRecorders[pageKey].timerId);
        delete activeRecorders[pageKey];

        recordBtn.textContent = "🎙 Record";
        recordBtn.classList.remove("btn-recording");

        const blob = new Blob(chunks, { type: mimeType });
        if (blob.size === 0) {
            showToast("Recording was empty — try again", "error");
            return;
        }

        const file = new File([blob], `${pageKey}_recording.webm`, { type: mimeType });
        fileLabel.textContent = `Recording (${formatDuration(Date.now() - startedAt)})`;
        uploadAndTranscribe(pageKey, file);
    };

    recorder.start();
    recordBtn.classList.add("btn-recording");
    recordBtn.textContent = "⏹ Stop (0:00)";

    const timerId = setInterval(() => {
        recordBtn.textContent = `⏹ Stop (${formatDuration(Date.now() - startedAt)})`;
    }, 500);

    activeRecorders[pageKey] = { recorder, timerId };
}

function formatDuration(ms) {
    const totalSec = Math.floor(ms / 1000);
    const min = Math.floor(totalSec / 60);
    const sec = totalSec % 60;
    return `${min}:${sec.toString().padStart(2, "0")}`;
}

function showPreview(pageKey, text) {
    const previewBox = document.getElementById(`preview-${pageKey}`);
    const previewTextarea = document.getElementById(`preview-text-${pageKey}`);
    if (!previewBox || !previewTextarea) return;

    previewTextarea.value = text;
    previewBox.classList.add("active");

    const confirmBtn = document.getElementById(`confirm-${pageKey}`);
    confirmBtn.onclick = () => confirmPage(pageKey, previewTextarea.value);

    const editBtn = document.getElementById(`edit-${pageKey}`);
    editBtn.onclick = () => openKeyboard(previewTextarea);
}

async function confirmPage(pageKey, editedText) {
    try {
        const res = await fetch(`/api/confirm/${pageKey}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: editedText })
        });
        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || "Confirm failed", "error");
            return;
        }

        showToast(`${pageKey} confirmed and filled`, "success");
        loadStatus();

    } catch (err) {
        showToast("Confirm failed: " + err.message, "error");
    }
}

async function skipPage(pageKey) {
    try {
        const res = await fetch(`/api/skip/${pageKey}`, { method: "POST" });
        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || "Skip failed", "error");
            return;
        }

        showToast(`${pageKey} skipped`, "");
        loadStatus();

    } catch (err) {
        showToast("Skip failed: " + err.message, "error");
    }
}

// ── Malayalam virtual keyboard ────────────────────────────────
const KB_VOWELS = ["അ", "ആ", "ഇ", "ഈ", "ഉ", "ഊ", "ഋ", "എ", "ഏ", "ഐ", "ഒ", "ഓ", "ഔ", "അം", "അഃ"];
const KB_CONSONANTS = ["ക", "ഖ", "ഗ", "ഘ", "ങ", "ച", "ഛ", "ജ", "ഝ", "ഞ", "ട", "ഠ", "ഡ", "ഢ", "ണ",
                        "ത", "ഥ", "ദ", "ധ", "ന", "പ", "ഫ", "ബ", "ഭ", "മ", "യ", "ര", "ല", "വ",
                        "ശ", "ഷ", "സ", "ഹ", "ള", "ഴ", "റ"];
const KB_MATRAS = ["ാ", "ി", "ീ", "ു", "ൂ", "ൃ", "െ", "േ", "ൈ", "ൊ", "ോ", "ൌ", "ൗ", "്", "ം", "ഃ"];
const KB_OTHERS = ["ൻ", "ർ", "ൽ", "ൾ", "ൺ", "ൿ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    ".", ",", "?", "!", "-"];

let activeTextarea = null;

function buildKeyboardGrid(containerId, chars) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";
    chars.forEach((ch) => {
        const key = document.createElement("button");
        key.type = "button";
        key.className = "kb-key";
        key.textContent = ch;
        key.addEventListener("click", () => insertChar(ch));
        container.appendChild(key);
    });
}

function initKeyboard() {
    buildKeyboardGrid("kb-vowels", KB_VOWELS);
    buildKeyboardGrid("kb-consonants", KB_CONSONANTS);
    buildKeyboardGrid("kb-matras", KB_MATRAS);
    buildKeyboardGrid("kb-others", KB_OTHERS);

    document.getElementById("kb-space").addEventListener("click", () => insertChar(" "));
    document.getElementById("kb-backspace").addEventListener("click", kbBackspace);
    document.getElementById("kb-done").addEventListener("click", closeKeyboard);
    document.getElementById("kb-close").addEventListener("click", closeKeyboard);

    makeDraggable(document.getElementById("kb-header"), document.getElementById("kb-modal"));
}

// Drag-to-reposition for the floating keyboard panel (mouse + touch)
function makeDraggable(handle, panel) {
    let dragging = false;
    let offsetX = 0;
    let offsetY = 0;

    function point(e) {
        if (e.touches && e.touches.length > 0) {
            return { x: e.touches[0].clientX, y: e.touches[0].clientY };
        }
        return { x: e.clientX, y: e.clientY };
    }

    function dragStart(e) {
        dragging = true;
        const rect = panel.getBoundingClientRect();
        const p = point(e);
        offsetX = p.x - rect.left;
        offsetY = p.y - rect.top;

        // Switch from the default right-anchored position to an
        // explicit left/top so dragging math stays consistent.
        panel.style.right = "auto";
        panel.style.left = `${rect.left}px`;
        panel.style.top = `${rect.top}px`;
    }

    function dragMove(e) {
        if (!dragging) return;
        const p = point(e);
        let newLeft = p.x - offsetX;
        let newTop = p.y - offsetY;

        const maxLeft = window.innerWidth - panel.offsetWidth;
        const maxTop = window.innerHeight - panel.offsetHeight;
        newLeft = Math.max(0, Math.min(newLeft, maxLeft));
        newTop = Math.max(0, Math.min(newTop, maxTop));

        panel.style.left = `${newLeft}px`;
        panel.style.top = `${newTop}px`;
        e.preventDefault();
    }

    function dragEnd() {
        dragging = false;
    }

    handle.addEventListener("mousedown", dragStart);
    document.addEventListener("mousemove", dragMove);
    document.addEventListener("mouseup", dragEnd);

    handle.addEventListener("touchstart", dragStart, { passive: true });
    document.addEventListener("touchmove", dragMove, { passive: false });
    document.addEventListener("touchend", dragEnd);
}

function openKeyboard(textarea) {
    activeTextarea = textarea;
    document.getElementById("kb-overlay").classList.add("active");
    textarea.focus();
}

function closeKeyboard() {
    document.getElementById("kb-overlay").classList.remove("active");
    activeTextarea = null;
}

function insertChar(ch) {
    if (!activeTextarea) return;
    const start = activeTextarea.selectionStart;
    const end = activeTextarea.selectionEnd;
    const val = activeTextarea.value;

    activeTextarea.value = val.slice(0, start) + ch + val.slice(end);
    const newPos = start + ch.length;
    activeTextarea.selectionStart = activeTextarea.selectionEnd = newPos;
    activeTextarea.focus();
}

function kbBackspace() {
    if (!activeTextarea) return;
    const start = activeTextarea.selectionStart;
    const end = activeTextarea.selectionEnd;
    const val = activeTextarea.value;

    if (start !== end) {
        activeTextarea.value = val.slice(0, start) + val.slice(end);
        activeTextarea.selectionStart = activeTextarea.selectionEnd = start;
    } else if (start > 0) {
        activeTextarea.value = val.slice(0, start - 1) + val.slice(start);
        activeTextarea.selectionStart = activeTextarea.selectionEnd = start - 1;
    }
    activeTextarea.focus();
}

// ── Landing screen + topbar: JSON upload, sample data ────────
async function handleJsonUpload(file) {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/api/upload_json", { method: "POST", body: formData });
        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || "Upload failed", "error");
            return;
        }

        showToast(`Loaded ${data.page_count} pages`, "success");
        await loadStatus();
        showScreen("screen-main");

    } catch (err) {
        showToast("Upload failed: " + err.message, "error");
    }
}

document.getElementById("json-upload-landing").addEventListener("change", (e) => {
    handleJsonUpload(e.target.files[0]);
});

document.getElementById("json-upload-topbar").addEventListener("change", (e) => {
    handleJsonUpload(e.target.files[0]);
});

document.getElementById("sample-data-btn").addEventListener("click", async () => {
    await loadStatus();
    showScreen("screen-main");
});

document.getElementById("save-btn").addEventListener("click", async () => {
    try {
        const res = await fetch("/api/save", { method: "POST" });
        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || "Save failed", "error");
            return;
        }

        showToast("Saved to data/output/", "success");

    } catch (err) {
        showToast("Save failed: " + err.message, "error");
    }
});

// ── Utility ───────────────────────────────────────────────────
function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// ── Init ──────────────────────────────────────────────────────
initKeyboard();
loadStatus();