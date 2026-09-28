"use strict";

const el = (id) => document.getElementById(id);
const state = {file: null, previewUrl: null, ready: false, busy: false, result: null, view: "overlay", classes: [], checking: false};
const allowedSuffix = /\.(jpe?g|png|webp|bmp)$/i;
const numberFormat = new Intl.NumberFormat("zh-CN");

function showError(message = "") {
  el("error").textContent = message;
  el("error").hidden = !message;
}
function updateControls() {
  el("run").disabled = !state.file || !state.ready || state.busy;
  el("clear").disabled = !state.file || state.busy;
  el("file").disabled = state.busy;
  el("drop").classList.toggle("busy", state.busy);
  el("runSpinner").hidden = !state.busy;
  el("runLabel").textContent = state.busy ? "正在上传与分析…" : "开始分析";
  el("retryHealth").disabled = state.busy || state.checking;
}
function resetResult() {
  state.result = null;
  el("results").hidden = true;
  el("empty").hidden = false;
  el("resultStatus").textContent = "等待分析";
  el("resultStatus").classList.remove("done");
  el("resultImage").removeAttribute("src");
  el("empty").classList.remove("working");
  el("empty").querySelector("h3").textContent = "从一张苹果叶图片开始";
}
function clearFile() {
  if (state.busy) return;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = null;
  state.previewUrl = null;
  el("file").value = "";
  el("preview").removeAttribute("src");
  el("preview").hidden = true;
  el("uploadPrompt").hidden = false;
  el("filename").textContent = "尚未选择图片";
  resetResult();
  el("resultStatus").textContent = "等待上传";
  showError();
  updateControls();
}
function chooseFile(file) {
  if (!file || state.busy) return;
  if (!allowedSuffix.test(file.name)) {
    showError("请选择 JPG、PNG、WEBP 或 BMP 图片。");
    return;
  }
  if (!file.size || file.size > 8 * 1024 * 1024) {
    showError(file.size ? "图片不能超过 8 MB，请压缩后再上传。" : "图片文件为空，请重新选择。");
    return;
  }
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = file;
  state.previewUrl = URL.createObjectURL(file);
  el("preview").src = state.previewUrl;
  el("preview").hidden = false;
  el("uploadPrompt").hidden = true;
  el("filename").textContent = file.name + " · " + (file.size / 1024 / 1024).toFixed(2) + " MB";
  resetResult();
  showError();
  updateControls();
}
el("preview").addEventListener("error", () => {
  if (!state.file) return;
  clearFile();
  showError("无法预览这张图片，请检查文件是否损坏。");
});
el("file").addEventListener("change", () => {
  chooseFile(el("file").files[0]);
  el("file").value = "";
});
el("clear").addEventListener("click", clearFile);
for (const name of ["dragenter", "dragover"]) {
  el("drop").addEventListener(name, (event) => {
    event.preventDefault();
    if (!state.busy) el("drop").classList.add("drag");
  });
}
for (const name of ["dragleave", "drop"]) {
  el("drop").addEventListener(name, (event) => {
    event.preventDefault();
    el("drop").classList.remove("drag");
  });
}
el("drop").addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));
// Prevent accidentally navigating away when a file is dropped outside the uploader.
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

async function checkHealth() {
  if (state.checking || state.busy) return;
  state.checking = true;
  state.ready = false;
  updateControls();
  el("serverState").textContent = "正在检查模型";
  try {
    const response = await fetch("/api/health", {signal: AbortSignal.timeout(30000)});
    if (!response.ok) throw new Error("后端状态请求失败");
    const data = await response.json();
    state.ready = data.status === "ready";
    state.classes = data.classes || [];
    el("serverState").textContent = state.ready ? "模型已就绪 · " + data.device.toUpperCase() : "模型未就绪";
    el("modelInfo").textContent = state.ready ? data.model_name + " · " + data.input_size + " × " + data.input_size + " 输入" : data.message;
    if (!state.ready) showError(data.message);
    else showError();
  } catch (error) {
    el("serverState").textContent = "服务未连接";
    el("modelInfo").textContent = "请启动后端服务，再点击重新检查";
    showError("无法连接后端，请检查启动终端，再点击右上角重新检查。");
  } finally {
    state.checking = false;
    el("statusDot").className = "status-dot " + (state.ready ? "ready" : "unavailable");
    updateControls();
  }
}
el("retryHealth").addEventListener("click", checkHealth);

function makeSwatch(color) {
  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.backgroundColor = color;
  return swatch;
}
function renderLegend() {
  el("legend").replaceChildren();
  for (const item of state.classes) {
    const label = document.createElement("span");
    label.className = "legend-item";
    label.append(makeSwatch(item.color), document.createTextNode(item.name));
    el("legend").append(label);
  }
}
function setView(view) {
  if (!state.result) return;
  state.view = view;
  el("resultImage").src = state.result[view];
  el("resultImage").alt = view === "overlay" ? "模型预测病斑叠加图" : "模型预测彩色分割掩膜";
  for (const item of ["overlay", "mask"]) {
    const active = view === item;
    el(item + "Tab").setAttribute("aria-selected", String(active));
    el(item + "Tab").tabIndex = active ? 0 : -1;
  }
  el("imagePanel").setAttribute("aria-labelledby", view + "Tab");
}
for (const view of ["overlay", "mask"]) {
  const tab = el(view + "Tab");
  tab.addEventListener("click", () => setView(view));
  tab.addEventListener("keydown", (event) => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      const next = event.key === "Home" ? "overlay" : event.key === "End" ? "mask" : view === "overlay" ? "mask" : "overlay";
      setView(next);
      el(next + "Tab").focus();
    }
  });
}
function renderResult(payload) {
  state.result = payload;
  const s = payload.summary;
  const noLeaf = s.status === "no_leaf";
  el("empty").hidden = true;
  el("results").hidden = false;
  el("resultStatus").textContent = "分析完成";
  el("resultStatus").classList.add("done");
  setView("overlay");
  renderLegend();
  el("dominant").textContent = s.most_predicted_disease;
  el("areaPercent").textContent = noLeaf ? "—" : s.total_disease_percent.toFixed(2) + "%";
  el("inferenceTime").textContent = (s.inference_ms / 1000).toFixed(2) + " s";
  el("dimensions").textContent = "图片 " + s.image_width + " × " + s.image_height;
  el("resultNotice").textContent = noLeaf
    ? "未识别到叶片，无法估计病斑比例。请换一张苹果叶主体更清晰的图片。"
    : s.status === "no_lesion"
      ? "模型本次未检出病斑；这不代表叶片一定健康。可以结合原图继续观察。"
      : payload.notice;
  el("pixelInfo").textContent = "预测叶片 " + numberFormat.format(s.predicted_leaf_pixels) + " px · 病斑 " + numberFormat.format(s.disease_pixels) + " px";
  el("diseaseAreas").replaceChildren();
  for (const item of payload.disease_areas) {
    const row = document.createElement("div");
    row.className = "disease-row";
    const name = document.createElement("span");
    name.className = "disease-name";
    name.append(makeSwatch(item.color), document.createTextNode(item.name));
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.backgroundColor = item.color;
    fill.style.width = Math.min(100, Math.max(0, item.percent_of_predicted_leaf)) + "%";
    track.append(fill);
    const value = document.createElement("strong");
    value.textContent = noLeaf ? "—" : item.percent_of_predicted_leaf.toFixed(2) + "%";
    row.title = item.name + "：" + numberFormat.format(item.pixels) + " 个预测像素";
    row.append(name, track, value);
    el("diseaseAreas").append(row);
  }
}
el("run").addEventListener("click", async () => {
  if (!state.file || !state.ready || state.busy) return;
  resetResult();
  state.busy = true;
  updateControls();
  showError();
  el("resultStatus").textContent = "正在分析";
  el("empty").classList.add("working");
  el("empty").querySelector("h3").textContent = "模型正在分析这张叶片…";
  try {
    const form = new FormData();
    form.append("file", state.file);
    const response = await fetch("/api/predict", {method: "POST", body: form, signal: AbortSignal.timeout(120000)});
    const payload = await response.json();
    if (!response.ok) {
      if (response.status === 503) state.ready = false;
      throw new Error(typeof payload.detail === "string" ? payload.detail : "请求失败，请重新选择图片。");
    }
    renderResult(payload);
  } catch (error) {
    resetResult();
    el("resultStatus").textContent = "分析失败";
    showError(error.name === "TimeoutError" ? "分析超时，请缩小图片后重试。" : error.message || "请求失败，请检查后端连接。");
    if (!state.ready) {
      el("serverState").textContent = "模型未就绪";
      el("statusDot").className = "status-dot unavailable";
    }
  } finally {
    state.busy = false;
    el("empty").classList.remove("working");
    updateControls();
  }
});
function download(href, name) {
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = name;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
}
function resultStem() {
  return state.file.name.replace(/\.[^.]+$/, "").replace(/[<>:"/\\|?*]/g, "_");
}
el("downloadImage").addEventListener("click", () => {
  if (state.result) download(state.result[state.view], resultStem() + "_" + state.view + ".png");
});
el("downloadReport").addEventListener("click", () => {
  if (!state.result) return;
  const {overlay, mask, label_mask, ...report} = state.result;
  const blob = new Blob([JSON.stringify(report, null, 2)], {type: "application/json;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  download(url, resultStem() + "_analysis.json");
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
window.addEventListener("beforeunload", () => {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
});
checkHealth();
