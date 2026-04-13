(function () {
  function apiBase() {
    const m = document.querySelector('meta[name="autods-api-base"]');
    const raw = m && m.getAttribute("content");
    if (!raw || !raw.trim()) return "";
    const t = raw.trim();
    if (t === "__AUTODS_API_BASE__") return "";
    return t.replace(/\/$/, "");
  }

  function apiUrl(path) {
    const base = apiBase();
    if (!path.startsWith("/")) path = "/" + path;
    return base ? base + path : path;
  }

  async function parseErrorResponse(res) {
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    if (ct.includes("application/json")) {
      try {
        return await res.json();
      } catch {
        return {};
      }
    }
    const text = await res.text();
    return { detail: text.slice(0, 400) || res.statusText };
  }

  function formatHttpError(res, data) {
    const d = data && data.detail;
    const base =
      d == null
        ? JSON.stringify(data || {})
        : Array.isArray(d)
          ? d.map((x) => x.msg || JSON.stringify(x)).join("; ")
          : String(d);
    if (res.status === 404 && (base === "Not Found" || !data.detail)) {
      return (
        'API returned 404 "Not Found". The page and API must use the same server.\n\n' +
        "Fix: stop Live Preview / file://, run from the project folder:\n" +
        "  python scripts\\run_web.py\n" +
        "Then open http://127.0.0.1:8000 in the browser.\n\n" +
        "If the UI is on another port on purpose, add to index.html:\n" +
        '  <meta name="autods-api-base" content="http://127.0.0.1:8000" />'
      );
    }
    return base;
  }

  const form1 = document.getElementById("form-step1");
  const useSample = document.getElementById("use_sample");
  const uploadWrap = document.getElementById("upload-wrap");
  const fileInput = document.getElementById("file");
  const btnPrepare = document.getElementById("btn-prepare");
  const btnBack2 = document.getElementById("btn-back-2");
  const btnRun = document.getElementById("btn-run");
  const btnBack3 = document.getElementById("btn-back-3");
  const targetSelect = document.getElementById("target_select");
  const targetCustom = document.getElementById("target_custom");
  const suggestionsEl = document.getElementById("suggestions");
  const prepareSummary = document.getElementById("prepare-summary");
  const liveMsg = document.getElementById("live-msg");
  const liveLog = document.getElementById("live-log");
  const pulse = document.getElementById("pulse");
  const errorCard = document.getElementById("error-card");
  const errorBody = document.getElementById("error-body");
  const answerCard = document.getElementById("answer-card");
  const answerBody = document.getElementById("answer-body");
  const jsonCard = document.getElementById("json-card");
  const jsonBody = document.getElementById("json-body");
  const csvCard = document.getElementById("csv-card");
  const csvDownload = document.getElementById("csv-download");
  const stepEls = document.querySelectorAll(".steps .step");
  const panels = document.querySelectorAll(".step-panel");

  let cachedUserQuery = "";

  function setStep(n) {
    stepEls.forEach((el) => {
      const s = el.dataset.step;
      el.classList.toggle("active", s === String(n));
      el.classList.toggle("done", Number(s) < n);
    });
    panels.forEach((p) => {
      p.classList.toggle("hidden", p.dataset.panel !== String(n));
    });
  }

  function toggleUpload() {
    if (useSample.checked) {
      uploadWrap.classList.add("disabled");
      fileInput.value = "";
    } else {
      uploadWrap.classList.remove("disabled");
    }
  }

  useSample.addEventListener("change", toggleUpload);
  toggleUpload();

  function buildFormDataForRun() {
    const fd = new FormData();
    fd.set("user_query", cachedUserQuery);
    fd.set("use_sample", useSample.checked ? "true" : "false");
    const custom = targetCustom.value.trim();
    const sel = targetSelect.value;
    const target = custom || sel;
    if (!target) {
      throw new Error("Choose or enter a target column.");
    }
    fd.set("target_column", target);
    if (!useSample.checked && fileInput.files[0]) {
      fd.append("file", fileInput.files[0]);
    }
    return fd;
  }

  function hidePostRun() {
    errorCard.classList.add("hidden");
    answerCard.classList.add("hidden");
    jsonCard.classList.add("hidden");
    csvCard.classList.add("hidden");
  }

  function fillTargetUI(data) {
    prepareSummary.textContent = `${data.row_count} rows, ${data.column_count} columns — pick what to predict.`;
    suggestionsEl.innerHTML = "";
    (data.suggestions || []).forEach((s) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "suggestion";
      b.innerHTML = `<strong>${escapeHtml(s.name)}</strong><span>${escapeHtml(s.rationale)}</span>`;
      b.addEventListener("click", () => {
        targetSelect.value = s.name;
        targetCustom.value = "";
      });
      suggestionsEl.appendChild(b);
    });
    targetSelect.innerHTML = "";
    (data.columns || []).forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      targetSelect.appendChild(opt);
    });
    const firstSug = data.suggestions && data.suggestions[0];
    if (firstSug) {
      targetSelect.value = firstSug.name;
    } else if (data.columns && data.columns[0]) {
      targetSelect.value = data.columns[0];
    }
    targetCustom.value = "";
  }

  function escapeHtml(t) {
    const d = document.createElement("div");
    d.textContent = t;
    return d.innerHTML;
  }

  form1.addEventListener("submit", async (e) => {
    e.preventDefault();
    hidePostRun();
    btnPrepare.disabled = true;
    const fd = new FormData(form1);
    fd.set("use_sample", useSample.checked ? "true" : "false");
    if (useSample.checked) {
      fd.delete("file");
    }
    cachedUserQuery = (fd.get("user_query") || "").toString().trim();

    try {
      const res = await fetch(apiUrl("/api/prepare"), { method: "POST", body: fd });
      const data = res.ok ? await res.json() : await parseErrorResponse(res);
      if (!res.ok) {
        throw new Error(formatHttpError(res, data));
      }
      fillTargetUI(data);
      setStep(2);
    } catch (err) {
      errorCard.classList.remove("hidden");
      errorBody.textContent = String(err);
      setStep(1);
    } finally {
      btnPrepare.disabled = false;
    }
  });

  btnBack2.addEventListener("click", () => {
    setStep(1);
    hidePostRun();
  });

  btnBack3.addEventListener("click", () => {
    liveLog.textContent = "";
    pulse.classList.remove("off");
    btnBack3.classList.add("hidden");
    setStep(1);
    hidePostRun();
  });

  targetCustom.addEventListener("input", () => {
    if (targetCustom.value.trim()) {
      targetSelect.selectedIndex = -1;
    }
  });

  targetSelect.addEventListener("change", () => {
    targetCustom.value = "";
  });

  async function readNdjsonStream(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalResult = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        let ev;
        try {
          ev = JSON.parse(line);
        } catch {
          continue;
        }
        if (ev.type === "progress") {
          const sm = ev.summary || {};
          const parts = [];
          if (sm.rows != null) parts.push(`${sm.rows}×${sm.cols}`);
          if (sm.profiled_columns != null) parts.push(`profile: ${sm.profiled_columns} cols`);
          if (sm.violations != null) parts.push(`violations: ${sm.violations}`);
          if (sm.cleaning_steps != null) parts.push(`clean steps: ${sm.cleaning_steps}`);
          if (sm.pass_count != null) parts.push(`pass: ${sm.pass_count}`);
          if (sm.has_model) parts.push("model ✓");
          if (sm.has_final_answer) parts.push("answer ✓");
          liveMsg.textContent = parts.length ? parts.join(" · ") : "Working…";
          if (ev.console_snippet) {
            liveLog.textContent = ev.console_snippet;
            liveLog.scrollTop = liveLog.scrollHeight;
          }
        } else if (ev.type === "complete") {
          finalResult = ev.result;
        } else if (ev.type === "error") {
          if (ev.console_log) {
            liveLog.textContent = ev.console_log;
          }
          throw new Error(ev.message || "Pipeline error");
        }
      }
    }
    return finalResult;
  }

  btnRun.addEventListener("click", async () => {
    hidePostRun();
    let fd;
    try {
      fd = buildFormDataForRun();
    } catch (err) {
      errorCard.classList.remove("hidden");
      errorBody.textContent = String(err);
      return;
    }

    setStep(3);
    liveLog.textContent = "";
    liveMsg.textContent = "Connecting…";
    pulse.classList.remove("off");
    btnRun.disabled = true;
    btnBack3.classList.add("hidden");

    try {
      const res = await fetch(apiUrl("/api/run/stream"), {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const errJson = await parseErrorResponse(res);
        throw new Error(formatHttpError(res, errJson));
      }

      const data = await readNdjsonStream(res);

      pulse.classList.add("off");
      liveMsg.textContent = "Finished.";
      btnBack3.classList.remove("hidden");

      if (data && data.console_log) {
        liveLog.textContent = data.console_log;
      }

      if (!data || !data.ok) {
        errorCard.classList.remove("hidden");
        errorBody.textContent = (data && data.error) || "Pipeline failed.";
        return;
      }

      if (data.final_answer) {
        answerCard.classList.remove("hidden");
        answerBody.textContent = data.final_answer;
      }

      const { console_log, clean_csv, ok, ...rest } = data;
      jsonCard.classList.remove("hidden");
      jsonBody.textContent = JSON.stringify(rest, null, 2);

      if (clean_csv) {
        csvCard.classList.remove("hidden");
        if (csvDownload._url) URL.revokeObjectURL(csvDownload._url);
        const blob = new Blob([clean_csv], { type: "text/csv;charset=utf-8" });
        csvDownload._url = URL.createObjectURL(blob);
        csvDownload.href = csvDownload._url;
        csvDownload.download = "cleaned_data.csv";
      }
    } catch (err) {
      pulse.classList.add("off");
      liveMsg.textContent = "Failed.";
      btnBack3.classList.remove("hidden");
      errorCard.classList.remove("hidden");
      errorBody.textContent = String(err);
    } finally {
      btnRun.disabled = false;
    }
  });
})();
