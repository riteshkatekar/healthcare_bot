// app.js

(() => {
  "use strict";

  if (window.__medicalChatbotAppInitialized) return;
  window.__medicalChatbotAppInitialized = true;

  const $ = (id) => document.getElementById(id);

  const el = {
    form: $("messageArea"),
    log: $("messageFormeight"),
    input: $("text"),
    imageInput: $("imageInput"),
    fileInput: $("fileInput"),
    thumbs: $("thumbsContainer"),
    filePreview: $("filePreview"),
    processing: $("processingList"),
    uploadBtn: $("uploadBtn"),
    fileBtn: $("fileBtn"),
    micBtn: $("micBtn"),
    clearBtn: $("clearBtn"),
    sendBtn: $("send"),
  };

  if (
    !el.form ||
    !el.log ||
    !el.input ||
    !el.imageInput ||
    !el.fileInput ||
    !el.thumbs ||
    !el.filePreview ||
    !el.uploadBtn ||
    !el.fileBtn ||
    !el.micBtn ||
    !el.clearBtn ||
    !el.sendBtn
  ) {
    console.error("Medical chatbot UI is missing required elements.");
    return;
  }

  const state = {
    images: [],
    file: null,
    sending: false,
    recognition: null,
    recognizing: false,
    recorder: null,
    recording: false,
    audioChunks: [],
    preparingMedia: 0,

    streamController: null,
    lastAssistantRow: null,
    lastAssistantText: "",
    lastAssistantLanguage: "en",
    speaking: null,
  };

  const MAX_IMAGES = 20;
  const MAX_ORIGINAL_UPLOAD_BYTES = 50 * 1024 * 1024;
  const MAX_RESIZED_IMAGE_BYTES = 10 * 1024 * 1024;

  let sendLock = false;
  let lastMessageHash = null;
  let lastSendTime = 0;

  const REGEN_ICON_SVG = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M3 12a9 9 0 0 1 15.3-6.4L20 8" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M20 4v4h-4" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M21 12a9 9 0 0 1-15.3 6.4L4 16" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M4 20v-4h4" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  `;

  function hashMessage(text, attachmentSignature) {
    return `${String(text || "").trim()}__${attachmentSignature}`;
  }

  function attachmentSignature() {
    const fileSig = state.file ? `${state.file.name}:${state.file.size}` : "";
    const imageSig = state.images
      .map((img) => {
        const sourceSize = img.original?.size || img.blob?.size || 0;
        return `${img.name || "image"}:${sourceSize}`;
      })
      .join("|");
    return `${fileSig}::${imageSig}`;
  }

  function nowTime() {
    const d = new Date();
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  }

  function scrollBottom() {
    el.log.scrollTop = el.log.scrollHeight;
  }

  function clearTyping() {
    const node = el.log.querySelector("[data-typing='1']");
    if (node) node.remove();
  }

  function setBusy(value) {
    state.sending = value;
    el.sendBtn.disabled = value;
    el.uploadBtn.disabled = value;
    el.fileBtn.disabled = value;
    el.micBtn.disabled = value;
    el.clearBtn.disabled = value;
  }

  function resetComposerInputs() {
    el.input.value = "";
    el.imageInput.value = "";
    el.fileInput.value = "";
    el.input.placeholder = "Type your message...";
  }

  function safeHtmlToText(value) {
    return String(value ?? "")
      .replace(/<[^>]*>/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function escapeHtml(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function renderMessageText(text) {
    const safe = escapeHtml(String(text || ""));
    return safe
      .replace(/(\*|-)\s+/g, "\n• ")
      .replace(/\n/g, "<br>")
      .replace(/(<br>\s*)+/g, "<br>");
  }

  function plainTextFromNode(node) {
    return String(node?.dataset?.rawText || node?.textContent || "").trim();
  }

  async function copyText(text) {
    const value = String(text || "");
    if (!value) return;

    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(value);
      return;
    }

    const ta = document.createElement("textarea");
    ta.value = value;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }

  function revokeAllImageUrls() {
    for (const item of state.images) {
      if (item.previewUrl) {
        try {
          URL.revokeObjectURL(item.previewUrl);
        } catch (_) {}
      }
    }
  }

  function clearAttachments() {
    revokeAllImageUrls();
    state.images = [];
    state.file = null;
    renderComposerImageThumbs();
    renderComposerFilePreview();
    updatePlaceholder();
  }

  function supportedDocument(file) {
    if (!file || !file.name) return false;
    const ext = file.name.toLowerCase().split(".").pop();
    return ["pdf", "docx", "txt", "md", "csv", "json", "html", "htm", "xml", "log"].includes(ext);
  }

  function isImageFile(file) {
    return file && file.type && file.type.startsWith("image/");
  }

  function isSecureVoiceContext() {
    const localhost =
      location.hostname === "localhost" ||
      location.hostname === "127.0.0.1" ||
      location.hostname === "[::1]";
    return window.isSecureContext || localhost;
  }

  function appendBubbleRow(role) {
    const row = document.createElement("div");
    row.className =
      role === "user"
        ? "d-flex justify-content-end mb-3 align-items-end"
        : "d-flex justify-content-start mb-3 align-items-start";
    return row;
  }

  function buildUserAvatar() {
    const avatarWrap = document.createElement("div");
    avatarWrap.className = "img_cont_msg";
    avatarWrap.style.marginLeft = "10px";
    avatarWrap.innerHTML =
      '<img src="https://i.ibb.co/d5b84Xw/Untitled-design.png" class="rounded-circle user_img_msg" alt="you" />';
    return avatarWrap;
  }

  function buildBotAvatar() {
    const avatarWrap = document.createElement("div");
    avatarWrap.className = "img_cont_msg";
    avatarWrap.style.marginRight = "10px";
    avatarWrap.innerHTML =
      '<img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" class="rounded-circle user_img_msg" alt="bot" />';
    return avatarWrap;
  }

  function appendAssistantBubble(text, { error = false } = {}) {
    const row = appendBubbleRow("assistant");

    const bubble = document.createElement("div");
    bubble.className = "msg_cotainer";
    bubble.style.maxWidth = "82%";

    const content = document.createElement("div");
    content.className = "assistant-message-text";
    content.style.wordBreak = "break-word";
    content.innerHTML = error ? renderMessageText(`⚠️ ${text}`) : renderMessageText(text);
    bubble.dataset.rawText = String(text || "");

    const timeSpan = document.createElement("span");
    timeSpan.className = "msg_time";
    timeSpan.textContent = nowTime();

    bubble.appendChild(content);
    bubble.appendChild(timeSpan);

    row.appendChild(buildBotAvatar());
    row.appendChild(bubble);
    el.log.appendChild(row);
    scrollBottom();

    return bubble;
  }

  function appendTypingBubble(message = "Thinking...") {
    clearTyping();
    const row = appendBubbleRow("assistant");
    row.dataset.typing = "1";

    const bubble = document.createElement("div");
    bubble.className = "msg_cotainer";
    bubble.style.maxWidth = "82%";

    const content = document.createElement("div");
    content.className = "assistant-message-text";
    content.textContent = message;

    const indicator = document.createElement("div");
    indicator.className = "streaming_indicator";
    indicator.innerHTML = "<span></span><span></span><span></span>";
    indicator.style.marginTop = "8px";

    const timeSpan = document.createElement("span");
    timeSpan.className = "msg_time";
    timeSpan.textContent = nowTime();

    bubble.appendChild(content);
    bubble.appendChild(indicator);
    bubble.appendChild(timeSpan);

    row.appendChild(buildBotAvatar());
    row.appendChild(bubble);
    el.log.appendChild(row);
    scrollBottom();
  }

  function makeAttachmentMeta(type, filename, status) {
    const meta = document.createElement("div");
    meta.style.display = "flex";
    meta.style.flexDirection = "column";
    meta.style.gap = "2px";
    meta.style.minWidth = "0";

    const title = document.createElement("div");
    title.style.fontWeight = "700";
    title.style.fontSize = "14px";
    title.style.lineHeight = "1.3";
    title.style.wordBreak = "break-word";
    title.textContent = filename || (type === "image" ? "Uploaded image" : "Uploaded file");

    const sub = document.createElement("div");
    sub.style.fontSize = "12px";
    sub.style.opacity = "0.9";
    sub.textContent = type === "image" ? "Image attachment" : "Document attachment";

    meta.appendChild(title);
    meta.appendChild(sub);

    if (status) {
      const st = document.createElement("div");
      st.style.fontSize = "11px";
      st.style.opacity = "0.85";
      st.style.marginTop = "4px";
      st.textContent = status;
      meta.appendChild(st);
    }

    return meta;
  }

  function renderUserAttachmentBubble({ text = "", images = [], file = null, status = "analyzing..." }) {
    const row = appendBubbleRow("user");
    const bubble = document.createElement("div");
    bubble.className = "msg_cotainer_send";
    bubble.style.maxWidth = "78%";

    if (text) {
      const textBlock = document.createElement("div");
      textBlock.className = "user-message-text";
      textBlock.style.marginBottom = images.length || file ? "8px" : "0";
      textBlock.innerHTML = renderMessageText(text);
      bubble.appendChild(textBlock);
    }

    if (images.length) {
      const grid = document.createElement("div");
      grid.className = "user-images-inline";

      for (const item of images) {
        const wrapper = document.createElement("div");
        wrapper.style.display = "flex";
        wrapper.style.flexDirection = "column";
        wrapper.style.gap = "6px";
        wrapper.style.alignItems = "flex-start";

        const thumb = document.createElement("img");
        thumb.src = item.previewUrl;
        thumb.className = "user-image-chat-thumb";
        thumb.alt = item.name || "uploaded image";

        const label = document.createElement("div");
        label.style.fontSize = "12px";
        label.style.opacity = "0.9";
        label.style.maxWidth = "100px";
        label.style.overflow = "hidden";
        label.style.textOverflow = "ellipsis";
        label.style.whiteSpace = "nowrap";
        label.textContent = item.name || "image";

        wrapper.appendChild(thumb);
        wrapper.appendChild(label);
        grid.appendChild(wrapper);
      }

      bubble.appendChild(grid);
    }

    if (file) {
      const fileChip = document.createElement("div");
      fileChip.className = "selected_file_chip";
      fileChip.style.marginBottom = "0";
      fileChip.style.background = "rgba(255,255,255,0.10)";
      fileChip.style.maxWidth = "100%";

      const icon = document.createElement("div");
      icon.style.width = "28px";
      icon.style.height = "28px";
      icon.style.borderRadius = "8px";
      icon.style.display = "flex";
      icon.style.alignItems = "center";
      icon.style.justifyContent = "center";
      icon.style.background = "rgba(0,0,0,0.18)";
      icon.innerHTML =
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M7 3h7l5 5v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z" stroke="white" stroke-width="2"/><path d="M14 3v6h6" stroke="white" stroke-width="2"/></svg>';

      const meta = makeAttachmentMeta("file", file.name, status);
      fileChip.appendChild(icon);
      fileChip.appendChild(meta);
      bubble.appendChild(fileChip);
    }

    const timeSpan = document.createElement("span");
    timeSpan.className = "msg_time_send";
    timeSpan.textContent = nowTime();
    bubble.appendChild(timeSpan);

    row.appendChild(bubble);
    row.appendChild(buildUserAvatar());
    el.log.appendChild(row);
    scrollBottom();

    return bubble;
  }

  function renderComposerImageThumbs() {
    el.thumbs.innerHTML = "";

    if (!state.images.length) {
      el.thumbs.style.display = "none";
      return;
    }

    el.thumbs.style.display = "flex";

    state.images.forEach((item, idx) => {
      const box = document.createElement("div");
      box.className = "thumb_item";

      const img = document.createElement("img");
      img.src = item.previewUrl;
      img.alt = item.name || `image-${idx + 1}`;

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "thumb_remove";
      remove.setAttribute("aria-label", "Remove image");
      remove.textContent = "×";
      remove.addEventListener("click", () => removeImageAt(idx));

      const caption = document.createElement("div");
      caption.className = "thumb_caption";
      caption.textContent = item.name || "";

      box.appendChild(img);
      box.appendChild(remove);
      box.appendChild(caption);
      el.thumbs.appendChild(box);
    });
  }

  function renderComposerFilePreview() {
    el.filePreview.innerHTML = "";

    if (!state.file) {
      el.filePreview.style.display = "none";
      return;
    }

    el.filePreview.style.display = "block";

    const chip = document.createElement("div");
    chip.className = "selected_file_chip";

    const kind = document.createElement("div");
    kind.style.fontSize = "12px";
    kind.style.opacity = "0.85";
    kind.textContent = "Document";

    const name = document.createElement("div");
    name.className = "chip_name";
    name.textContent = state.file.name;

    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "chip_remove";
    rm.setAttribute("aria-label", "Remove document");
    rm.textContent = "×";
    rm.addEventListener("click", () => {
      state.file = null;
      renderComposerFilePreview();
      updatePlaceholder();
    });

    chip.appendChild(kind);
    chip.appendChild(name);
    chip.appendChild(rm);
    el.filePreview.appendChild(chip);
  }

  function removeImageAt(index) {
    if (index < 0 || index >= state.images.length) return;
    const item = state.images[index];
    if (item && item.previewUrl) {
      try {
        URL.revokeObjectURL(item.previewUrl);
      } catch (_) {}
    }
    state.images.splice(index, 1);
    renderComposerImageThumbs();
    updatePlaceholder();
  }

  async function resizeImageFile(file, maxWidth = 1400, quality = 0.82) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onerror = () => reject(new Error("Could not read image."));
      reader.onload = (ev) => {
        const img = new Image();

        img.onload = () => {
          const w = img.width || 0;
          const h = img.height || 0;
          if (!w || !h) return reject(new Error("Invalid image."));

          const scale = Math.min(1, maxWidth / w);
          const outW = Math.max(1, Math.round(w * scale));
          const outH = Math.max(1, Math.round(h * scale));

          const canvas = document.createElement("canvas");
          canvas.width = outW;
          canvas.height = outH;

          const ctx = canvas.getContext("2d");
          ctx.drawImage(img, 0, 0, outW, outH);

          canvas.toBlob((blob) => {
            if (!blob) return reject(new Error("Image conversion failed."));
            resolve(blob);
          }, "image/jpeg", quality);
        };

        img.onerror = () => reject(new Error("Could not decode image."));
        img.src = ev.target.result;
      };

      reader.readAsDataURL(file);
    });
  }

  function chooseSpeechLanguage() {
    const lang = (navigator.language || "en-IN").toLowerCase();
    if (lang.startsWith("hi")) return "hi-IN";
    if (lang.startsWith("mr")) return "mr-IN";
    return "en-IN";
  }

  function detectLanguageFromText(text) {
    const t = String(text || "").trim();
    if (!t) return "en-IN";

    if (/[\u0C80-\u0CFF]/.test(t)) return "kn-IN"; // Kannada
    if (/[\u0C00-\u0C7F]/.test(t)) return "te-IN"; // Telugu
    if (/[\u0B80-\u0BFF]/.test(t)) return "ta-IN"; // Tamil
    if (/[\u0D00-\u0D7F]/.test(t)) return "ml-IN"; // Malayalam
    if (/[\u0980-\u09FF]/.test(t)) return "bn-IN"; // Bengali
    if (/[\u0A80-\u0AFF]/.test(t)) return "gu-IN"; // Gujarati
    if (/[\u0A00-\u0A7F]/.test(t)) return "pa-IN"; // Punjabi

    if (/[\u0900-\u097F]/.test(t)) {
      const marathiHints = [
        "आहे", "काय", "का", "नाही", "मला", "सांग", "सांगू", "ताप", "खोकला", "दुखत", "औषध",
      ];
      const hasMarathiHint = marathiHints.some((w) => t.includes(w));
      return hasMarathiHint ? "mr-IN" : "hi-IN";
    }

    return "en-IN";
  }

  function pickVoiceForText(text) {
    const voices = window.speechSynthesis ? window.speechSynthesis.getVoices() : [];
    const targetLang = detectLanguageFromText(text);

    if (!voices || !voices.length) {
      return { lang: targetLang, voice: null };
    }

    const normalizedTarget = targetLang.toLowerCase();
    const exact = voices.find((v) => String(v.lang || "").toLowerCase() === normalizedTarget);
    if (exact) return { lang: exact.lang, voice: exact };

    const base = normalizedTarget.split("-")[0];
    const sameBase = voices.find((v) => String(v.lang || "").toLowerCase().startsWith(base));
    if (sameBase) return { lang: sameBase.lang, voice: sameBase };

    const englishFallback = voices.find((v) => String(v.lang || "").toLowerCase().startsWith("en"));
    if (englishFallback) return { lang: englishFallback.lang, voice: englishFallback };

    return { lang: targetLang, voice: voices[0] || null };
  }

  function stopSpeaking() {
    try {
      window.speechSynthesis.cancel();
    } catch (_) {}
    state.speaking = null;
  }

  function toggleVoiceReply({ bubble, button }) {
    const text = plainTextFromNode(bubble).trim();
    if (!text || !window.speechSynthesis || !window.SpeechSynthesisUtterance) return;

    if (state.speaking && state.speaking.bubble === bubble) {
      if (window.speechSynthesis.speaking && !window.speechSynthesis.paused) {
        window.speechSynthesis.pause();
        button.textContent = "▶";
        return;
      }

      if (window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
        button.textContent = "⏸";
        return;
      }
    }

    stopSpeaking();

    const utterance = new SpeechSynthesisUtterance(text);
    const picked = pickVoiceForText(text);

    utterance.lang = picked.lang;
    if (picked.voice) utterance.voice = picked.voice;

    utterance.onstart = () => {
      state.speaking = { bubble, utterance, button };
      button.textContent = "⏸";
    };

    utterance.onend = () => {
      if (state.speaking && state.speaking.button === button) {
        state.speaking = null;
        button.textContent = "🔊";
      }
    };

    utterance.onerror = () => {
      if (state.speaking && state.speaking.button === button) {
        state.speaking = null;
        button.textContent = "🔊";
      }
    };

    window.speechSynthesis.speak(utterance);
  }

  function makeToolButton(content, title, onClick, { html = false } = {}) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "msg_action_btn";
    btn.title = title;
    btn.setAttribute("aria-label", title);
    if (html) btn.innerHTML = content;
    else btn.textContent = content;
    btn.addEventListener("click", onClick);
    return btn;
  }

  function appendAssistantMessage({
    text = "",
    followups = [],
    canRegenerate = false,
    isStreaming = false,
  } = {}) {
    const row = appendBubbleRow("assistant");

    const bubble = document.createElement("div");
    bubble.className = "msg_cotainer";
    bubble.style.maxWidth = "82%";
    bubble.style.position = "relative";

    const topTools = document.createElement("div");
    topTools.className = "assistant-tools-top";

    const content = document.createElement("div");
    content.className = "assistant-message-text";
    content.style.wordBreak = "break-word";
    content.innerHTML =
      isStreaming && !text
        ? '<span class="streaming_indicator"><span></span><span></span><span></span></span>'
        : renderMessageText(text);

    bubble.dataset.rawText = String(text || "");

    const bottomTools = document.createElement("div");
    bottomTools.className = "assistant-tools-bottom";

    const followupsWrap = document.createElement("div");
    followupsWrap.className = "followups-wrap";

    const copyBtn = makeToolButton("⧉", "Copy response", async () => {
      try {
        const raw = plainTextFromNode(bubble);
        await copyText(raw);
        const old = copyBtn.textContent;
        copyBtn.textContent = "Copied ✓";
        setTimeout(() => {
          copyBtn.textContent = old;
        }, 1000);
      } catch (_) {}
    });

    const speakBtn = makeToolButton("🔊", "Play / pause voice reply", () => {
      toggleVoiceReply({ bubble, button: speakBtn });
    });

    topTools.appendChild(copyBtn);
    topTools.appendChild(speakBtn);

    let regenBtn = null;
    if (canRegenerate) {
      regenBtn = makeToolButton(REGEN_ICON_SVG, "Regenerate response", async () => {
        await regenerateLastResponse();
      }, { html: true });

      regenBtn.style.display = "inline-flex";
      regenBtn.style.alignItems = "center";
      regenBtn.style.justifyContent = "center";
      regenBtn.style.padding = "6px 8px";
      regenBtn.style.minWidth = "36px";
      regenBtn.style.minHeight = "32px";

      bottomTools.appendChild(regenBtn);
    }

    function renderFollowups(items) {
      followupsWrap.innerHTML = "";
      const questions = Array.isArray(items) ? items : [];
      for (const q of questions) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "followup_chip";
        chip.textContent = q;
        chip.addEventListener("click", async () => {
          el.input.value = q;
          el.input.focus();
          await handleSend();
        });
        followupsWrap.appendChild(chip);
      }
    }

    if (!isStreaming) {
      renderFollowups(followups);
    }

    const timeSpan = document.createElement("span");
    timeSpan.className = "msg_time";
    timeSpan.textContent = nowTime();

    bubble.appendChild(topTools);
    bubble.appendChild(content);
    bubble.appendChild(bottomTools);
    bubble.appendChild(followupsWrap);
    bubble.appendChild(timeSpan);

    row.appendChild(buildBotAvatar());
    row.appendChild(bubble);
    el.log.appendChild(row);
    scrollBottom();

    return {
      row,
      bubble,
      content,
      topTools,
      bottomTools,
      followupsWrap,
      setText(nextText) {
        bubble.dataset.rawText = String(nextText || "");
        content.innerHTML = renderMessageText(nextText);
        scrollBottom();
      },
      finalize({ answer = "", followups: items = [], canRegenerate: regen = true } = {}) {
        bubble.dataset.rawText = String(answer || "");
        content.innerHTML = renderMessageText(answer);

        if (regen && !regenBtn) {
          regenBtn = makeToolButton(REGEN_ICON_SVG, "Regenerate response", async () => {
            await regenerateLastResponse();
          }, { html: true });

          regenBtn.style.display = "inline-flex";
          regenBtn.style.alignItems = "center";
          regenBtn.style.justifyContent = "center";
          regenBtn.style.padding = "6px 8px";
          regenBtn.style.minWidth = "36px";
          regenBtn.style.minHeight = "32px";

          bottomTools.appendChild(regenBtn);
        }

        followupsWrap.innerHTML = "";
        const questions = Array.isArray(items) ? items : [];
        for (const q of questions) {
          const chip = document.createElement("button");
          chip.type = "button";
          chip.className = "followup_chip";
          chip.textContent = q;
          chip.addEventListener("click", async () => {
            el.input.value = q;
            el.input.focus();
            await handleSend();
          });
          followupsWrap.appendChild(chip);
        }

        state.lastAssistantRow = row;
        state.lastAssistantText = String(answer || "");
        scrollBottom();
      },
      destroy() {
        row.remove();
      },
    };
  }

  async function postForm(url, fd) {
    const res = await fetch(url, {
      method: "POST",
      body: fd,
      headers: {
        Accept: "application/json, text/event-stream",
      },
    });

    const contentType = res.headers.get("content-type") || "";
    if (contentType.includes("text/event-stream")) {
      return res;
    }

    let data = null;
    if (contentType.includes("application/json")) {
      try {
        data = await res.json();
      } catch (_) {
        data = null;
      }
    }

    if (!data) {
      const raw = await res.text().catch(() => "");
      data = { error: raw };
    }

    if (!res.ok) {
      const raw = safeHtmlToText(data.error || data.message || data.detail || `Request failed (${res.status})`);
      throw new Error(raw || `Request failed (${res.status})`);
    }

    return data;
  }

  function buildPayloadFD(message) {
    const fd = new FormData();
    const text = message || "";

    fd.append("message", text);
    fd.append("msg", text);
    fd.append("caption", text);
    fd.append("text", text);

    if (state.file) {
      fd.append("file", state.file, state.file.name);
    }

    for (const item of state.images) {
      if (item.blob) {
        fd.append("images", item.blob, item.name || "image.jpg");
      }
    }

    return fd;
  }

  function showMicSecurityHint() {
    appendAssistantBubble(
      "Microphone access needs HTTPS or localhost. Open this site on https:// or use localhost during development to use speech input.",
      { error: true }
    );
  }

  async function toggleVoiceInput() {
    if (state.sending) return;

    try {
      if (!isSecureVoiceContext()) {
        showMicSecurityHint();
        return;
      }

      const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;

      if (SpeechRecognitionCtor) {
        if (!state.recognition) {
          const recognition = new SpeechRecognitionCtor();
          recognition.lang = chooseSpeechLanguage();
          recognition.interimResults = false;
          recognition.continuous = false;
          recognition.maxAlternatives = 1;

          recognition.onresult = (event) => {
            const transcript = Array.from(event.results)
              .map((r) => (r[0] && r[0].transcript ? r[0].transcript : ""))
              .join("")
              .trim();

            if (transcript) {
              el.input.value = (el.input.value ? `${el.input.value} ` : "") + transcript;
              el.input.focus();
            }
          };

          recognition.onerror = (event) => {
            const err = event && event.error ? event.error : "unknown";
            appendAssistantBubble(`Voice input failed: ${err}`, { error: true });
          };

          recognition.onend = () => {
            state.recognizing = false;
            el.micBtn.classList.remove("active");
          };

          state.recognition = recognition;
        }

        if (!state.recognizing) {
          state.recognizing = true;
          el.micBtn.classList.add("active");
          state.recognition.start();
        } else {
          state.recognition.stop();
        }
        return;
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
        appendAssistantBubble("This browser does not support voice input.", { error: true });
        return;
      }

      if (state.recording && state.recorder) {
        state.recorder.stop();
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      state.recorder = recorder;
      state.audioChunks = [];

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) state.audioChunks.push(e.data);
      };

      recorder.onstop = async () => {
        const blob = new Blob(state.audioChunks, { type: "audio/webm" });
        const fd = new FormData();
        fd.append("audio", blob, "voice.webm");

        appendTypingBubble("Transcribing...");
        setBusy(true);

        try {
          const data = await postForm("/transcribe", fd);
          clearTyping();

          const transcript = (data && data.transcript ? String(data.transcript) : "").trim();
          if (transcript) {
            el.input.value = (el.input.value ? `${el.input.value} ` : "") + transcript;
            el.input.focus();
          }
        } catch (err) {
          clearTyping();
          appendAssistantBubble(err.message || "Transcription failed.", { error: true });
        } finally {
          try {
            stream.getTracks().forEach((track) => track.stop());
          } catch (_) {}
          setBusy(false);
          state.recording = false;
          el.micBtn.classList.remove("active");
        }
      };

      recorder.start();
      state.recording = true;
      el.micBtn.classList.add("active");
      appendAssistantBubble("Recording... click the mic again to stop.");
    } catch (err) {
      state.recognizing = false;
      state.recording = false;
      el.micBtn.classList.remove("active");
      appendAssistantBubble(err.message || "Voice input error.", { error: true });
    }
  }

  async function moveImagesToState(files) {
    const items = Array.from(files || []).filter(Boolean);
    if (!items.length) return;

    if (state.images.length + items.length > MAX_IMAGES) {
      appendAssistantBubble(`You can upload up to ${MAX_IMAGES} images at a time.`, { error: true });
      return;
    }

    for (const file of items) {
      if (!isImageFile(file)) continue;

      if (file.size > MAX_ORIGINAL_UPLOAD_BYTES) {
        appendAssistantBubble(`Image "${file.name}" is too large. Please choose a smaller file.`, {
          error: true,
        });
        continue;
      }

      state.preparingMedia += 1;
      const entry = {
        original: file,
        blob: null,
        name: file.name || "image.jpg",
        previewUrl: "",
      };

      state.images.push(entry);
      renderComposerImageThumbs();
      updatePlaceholder();

      try {
        const resized = await resizeImageFile(file);
        if (resized.size > MAX_RESIZED_IMAGE_BYTES) {
          throw new Error("Resized image is still too large.");
        }

        entry.blob = resized;

        if (entry.previewUrl) {
          try {
            URL.revokeObjectURL(entry.previewUrl);
          } catch (_) {}
        }

        entry.previewUrl = URL.createObjectURL(resized);
      } catch (_) {
        const idx = state.images.indexOf(entry);
        if (idx >= 0) state.images.splice(idx, 1);
        appendAssistantBubble(`Could not prepare "${file.name}".`, { error: true });
      } finally {
        state.preparingMedia = Math.max(0, state.preparingMedia - 1);
        renderComposerImageThumbs();
        updatePlaceholder();
      }
    }
  }

  function updatePlaceholder() {
    const imageCount = state.images.filter((x) => x.blob).length;
    const hasFile = !!state.file;

    if (imageCount && hasFile) {
      el.input.placeholder = `Uploaded ${imageCount} image${imageCount > 1 ? "s" : ""} and ${state.file.name} — add a question...`;
      return;
    }

    if (imageCount) {
      el.input.placeholder = `Uploaded ${imageCount} image${imageCount > 1 ? "s" : ""} — add a caption or question...`;
      return;
    }

    if (hasFile) {
      el.input.placeholder = `Uploaded ${state.file.name} — ask a question about it...`;
      return;
    }

    el.input.placeholder = "Type your message...";
  }

  function renderOutgoingUserMessage(message) {
    const images = state.images.filter((x) => x.blob);
    const file = state.file;
    const hasImg = images.length > 0;
    const hasFile = !!file;

    if (!message && !hasImg && !hasFile) return;

    renderUserAttachmentBubble({
      text: message || "",
      images: hasImg ? images : [],
      file: hasFile ? file : null,
      status: hasImg || hasFile ? "analyzing..." : "",
    });
  }

  function streamForm(url, fd, onToken, onDone) {
    const controller = new AbortController();
    state.streamController = controller;

    return fetch(url, {
      method: "POST",
      body: fd,
      signal: controller.signal,
      headers: {
        Accept: "text/event-stream",
      },
    }).then(async (res) => {
      if (!res.ok || !res.body) {
        const raw = await res.text().catch(() => "");
        throw new Error(raw || `Request failed (${res.status})`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          const rawEvent = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);

          const dataLines = rawEvent
            .split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.replace(/^data:\s*/, ""));

          const payloadText = dataLines.join("\n").trim();
          if (payloadText) {
            const payload = JSON.parse(payloadText);

            if (payload.type === "token") {
              onToken?.(payload.text || "");
            } else if (payload.type === "done") {
              onDone?.(payload);
            } else if (payload.type === "error") {
              throw new Error(payload.error || "Request failed.");
            }
          }

          boundary = buffer.indexOf("\n\n");
        }
      }

      state.streamController = null;
    });
  }

  async function handleSend() {
    const message = (el.input.value || "").trim();
    const hasAnyAttachment = state.preparingMedia > 0 || !!state.file || state.images.some((x) => x.blob);

    if (!message && !hasAnyAttachment) return;
    if (sendLock || window.__sendingNow) return;

    if (state.preparingMedia > 0) {
      appendAssistantBubble("Please wait for the uploaded image to finish preparing.", { error: true });
      return;
    }

    const now = Date.now();
    if (now - lastSendTime < 700) return;

    const currentHash = hashMessage(message, attachmentSignature());
    if (currentHash === lastMessageHash) return;

    sendLock = true;
    window.__sendingNow = true;
    lastSendTime = now;
    lastMessageHash = currentHash;

    setBusy(true);
    renderOutgoingUserMessage(message);

    const assistantUI = appendAssistantMessage({
      text: "",
      followups: [],
      canRegenerate: false,
      isStreaming: true,
    });

    try {
      const fd = hasAnyAttachment ? buildPayloadFD(message) : new FormData();
      const endpoint = hasAnyAttachment ? "/chat" : "/get";
      fd.append("stream", "1");

      if (!hasAnyAttachment) {
        fd.append("msg", message);
        fd.append("message", message);
        fd.append("caption", message);
        fd.append("text", message);
      }

      let accumulated = "";
      let finalPayload = null;

      await streamForm(
        endpoint,
        fd,
        (token) => {
          accumulated += token;
          clearTyping();
          assistantUI.setText(accumulated || " ");
        },
        (payload) => {
          finalPayload = payload;
        }
      );

      clearTyping();

      const answer = String(finalPayload?.answer || accumulated || "No response").trim();
      assistantUI.finalize({
        answer,
        followups: finalPayload?.followups || [],
        canRegenerate: true,
      });

      if (finalPayload?.emergency) {
        const banner = document.createElement("div");
        banner.className = "emergency-banner";
        banner.textContent = "⚠️ Emergency: Seek immediate medical help.";
        el.log.appendChild(banner);
        scrollBottom();
      }

      clearAttachments();
      resetComposerInputs();
    } catch (err) {
      clearTyping();
      assistantUI.destroy();
      appendAssistantBubble(err.message || "Something went wrong.", { error: true });
    } finally {
      setBusy(false);
      window.__sendingNow = false;
      sendLock = false;
      state.streamController = null;
    }
  }

  async function regenerateLastResponse() {
    if (sendLock || window.__sendingNow) return;

    if (state.streamController) {
      try {
        state.streamController.abort();
      } catch (_) {}
    }

    if (!state.lastAssistantRow) return;

    state.lastAssistantRow.remove();
    state.lastAssistantRow = null;

    sendLock = true;
    window.__sendingNow = true;
    setBusy(true);

    const assistantUI = appendAssistantMessage({
      text: "",
      followups: [],
      canRegenerate: false,
      isStreaming: true,
    });

    try {
      const fd = new FormData();
      fd.append("stream", "1");

      let accumulated = "";
      let finalPayload = null;

      await streamForm(
        "/regenerate",
        fd,
        (token) => {
          accumulated += token;
          assistantUI.setText(accumulated || " ");
        },
        (payload) => {
          finalPayload = payload;
        }
      );

      const answer = String(finalPayload?.answer || accumulated || "No response").trim();
      assistantUI.finalize({
        answer,
        followups: finalPayload?.followups || [],
        canRegenerate: true,
      });
    } catch (err) {
      assistantUI.destroy();
      appendAssistantBubble(err.message || "Regeneration failed.", { error: true });
    } finally {
      setBusy(false);
      window.__sendingNow = false;
      sendLock = false;
      state.streamController = null;
    }
  }

  function installEvents() {
    el.form.addEventListener("submit", (event) => {
      event.preventDefault();
      handleSend();
    });

    el.uploadBtn.addEventListener("click", () => {
      if (state.sending) return;
      el.imageInput.click();
    });

    el.fileBtn.addEventListener("click", () => {
      if (state.sending) return;
      el.fileInput.click();
    });

    el.micBtn.addEventListener("click", () => {
      toggleVoiceInput();
    });

    el.clearBtn.addEventListener("click", async () => {
      if (state.sending) return;

      try {
        if (state.streamController) {
          state.streamController.abort();
        }
      } catch (_) {}

      try {
        await fetch("/clear", { method: "POST" });
      } catch (_) {}

      el.log.innerHTML = "";
      clearAttachments();
      resetComposerInputs();
      lastMessageHash = null;
      lastSendTime = 0;
      sendLock = false;
      window.__sendingNow = false;
      state.lastAssistantRow = null;
      state.lastAssistantText = "";
      stopSpeaking();
      appendAssistantBubble("Conversation cleared.");
    });

    el.imageInput.addEventListener("change", (event) => {
      const files = event.target.files ? Array.from(event.target.files) : [];
      void moveImagesToState(files);
      el.imageInput.value = "";
      updatePlaceholder();
    });

    el.fileInput.addEventListener("change", (event) => {
      const file = event.target.files && event.target.files[0] ? event.target.files[0] : null;
      if (!file) return;

      if (!supportedDocument(file)) {
        appendAssistantBubble("Unsupported document type.", { error: true });
        el.fileInput.value = "";
        return;
      }

      state.file = file;
      renderComposerFilePreview();
      updatePlaceholder();

      el.fileInput.value = "";
      el.input.focus();
    });

    document.addEventListener("paste", (event) => {
      const items = event.clipboardData && event.clipboardData.items ? Array.from(event.clipboardData.items) : [];
      const files = items
        .filter((item) => item.kind === "file" && item.type && item.type.startsWith("image/"))
        .map((item) => item.getAsFile())
        .filter(Boolean);

      if (files.length) {
        void moveImagesToState(files);
        event.preventDefault();
      }
    });
  }

  function init() {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.onvoiceschanged = () => {
        window.speechSynthesis.getVoices();
      };
    }

    installEvents();
    updatePlaceholder();

    setTimeout(() => {
      el.input.focus();
    }, 200);
  }

  init();

  window.addEventListener("beforeunload", () => {
    try {
      if (state.streamController) state.streamController.abort();
    } catch (_) {}
    stopSpeaking();
  });
})();