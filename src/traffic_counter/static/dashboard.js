(function() {
"use strict";
var POLL_MS = 1000;
var STATUS_URL = "/api/status";
var SUMMARY_URL = "/api/summary";
var TIMESERIES_URL = "/api/timeseries";
var numberFormat = new Intl.NumberFormat("id-ID");
var lastGood = null;
var pending = false;
var timer = null;
function byId(id) {
return document.getElementById(id);
}
function formatInt(value) {
if (value === null || value === undefined) {
return "—";
}
var num = Number(value);
if (!isFinite(num)) {
return "—";
}
return numberFormat.format(Math.round(num));
}
function formatRate(value) {
if (value === null || value === undefined) {
return "—";
}
var num = Number(value);
if (!isFinite(num)) {
return "—";
}
return numberFormat.format(Math.round(num * 10) / 10);
}
function formatUnit(value, unit, formatter) {
var text = formatter(value);
if (text === "—") {
return "—";
}
return text + " " + unit;
}
function setElText(el, text) {
if (el && el.textContent !== text) {
el.textContent = text;
}
}
function setText(id, text) {
setElText(byId(id), text);
}
function shapeForState(state) {
if (state === "running" || state === "connected" || state === "live") {
return "●";
}
if (state === "degraded" || state === "connecting") {
return "▲";
}
if (state === "error" || state === "stopped" || state === "disconnected") {
return "■";
}
return "?";
}
function paintState(id, stateText) {
var el = byId(id);
if (!el) {
return;
}
el.setAttribute("data-state", stateText);
var glyph = shapeForState(stateText);
var shape = el.querySelector(".shape");
if (shape && shape.textContent !== glyph) {
shape.textContent = glyph;
}
var label = el.querySelector(".label");
if (label && label.textContent !== stateText) {
label.textContent = stateText;
}
if (!shape && !label) {
var wanted = glyph + " " + stateText;
if (el.textContent !== wanted) {
el.textContent = wanted;
}
}
}
function describeStatus(status) {
var appText = String(status.state || "unknown");
var streamText = String(status.stream_state || status.streamState || "unknown");
paintState("app-state", appText);
paintState("stream-state", streamText);
setText("actual-fps", formatRate(status.actual_fps));
setText("snapshot-age", formatUnit(status.snapshot_age_seconds, "dtk", formatInt));
setText("last-updated", String(status.generated_at || "—"));
}
function validSummary(summary) {
if (!summary) {
return false;
}
if (!summary.periods) {
return false;
}
if (!summary.periods.pagi || !summary.periods.siang || !summary.periods.sore) {
return false;
}
return true;
}
function renderSummary(summary) {
if (!validSummary(summary)) {
return false;
}
var host = byId("period-cards");
if (!host) {
return false;
}
var names = ["pagi", "siang", "sore"];
for (var i = 0; i < names.length; i = i + 1) {
var key = names[i];
var entry = summary.periods[key];
if (!entry) {
continue;
}
var card = host.querySelector('[data-period="' + key + '"]');
if (!card) {
continue;
}
var totalEl = card.querySelector('[data-role="total"]');
if (totalEl) {
setElText(totalEl, formatInt(entry.total));
}
var rateEl = card.querySelector('[data-role="rate"]');
if (rateEl) {
setElText(rateEl, formatUnit(entry.average_rate, "per mnt", formatRate));
}
card.setAttribute("data-state", entry.started ? "started" : "waiting");
}
return true;
}
function escapeText(value) {
return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function renderTimeseries(payload) {
if (!payload || !payload.minutes) {
return false;
}
var minutes = payload.minutes;
var chart = byId("traffic-chart");
if (chart) {
if (minutes.length === 0) {
chart.innerHTML = '<svg class="chart" width="880" height="160" role="img"><line x1="0" y1="150" x2="880" y2="150" stroke="rgb(90,100,120)" stroke-width="1"></line><text x="12" y="80" fill="rgb(200,210,225)">no data</text></svg>';
} else {
var peak = 1;
var k = 0;
for (k = 0; k < minutes.length; k = k + 1) {
var c = Number(minutes[k].count) || 0;
if (c > peak) {
peak = c;
}
}
var width = 880;
var base = 150;
var usable = 140;
var step = width / minutes.length;
var bar = step * 0.7;
var parts = [];
parts.push('<svg class="chart" width="880" height="160" role="img">');
parts.push('<line x1="0" y1="150" x2="880" y2="150" stroke="rgb(90,100,120)" stroke-width="1"></line>');
var dots = [];
for (var j = 0; j < minutes.length; j = j + 1) {
var count = Number(minutes[j].count) || 0;
var h = count / peak * usable;
var x = j * step + (step - bar) / 2;
var y = base - h;
parts.push('<rect x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '" width="' + bar.toFixed(1) + '" height="' + h.toFixed(1) + '" fill="rgb(255,180,60)"></rect>');
dots.push((x + bar / 2).toFixed(1) + "," + y.toFixed(1));
}
parts.push('<polyline points="' + dots.join(" ") + '" fill="none" stroke="rgb(120,200,255)" stroke-width="2"></polyline>');
parts.push("</svg>");
chart.innerHTML = parts.join("");
}
}
var table = byId("timeseries-table");
if (table) {
var body = table.querySelector("tbody");
if (body) {
var rows = [];
for (var m = 0; m < minutes.length; m = m + 1) {
var item = minutes[m];
rows.push("<tr><td>" + escapeText(item.minute_start) + "</td><td>" + escapeText(item.count) + "</td><td>" + escapeText(item.observed_seconds) + "</td><td>" + escapeText(item.period) + "</td></tr>");
}
body.innerHTML = rows.join("");
}
}
return true;
}
function markStale(message) {
var el = byId("data-warning");
if (el) {
el.setAttribute("data-state", "stale");
el.textContent = "STALE ▲ " + message + " — menampilkan data terakhir yang valid";
}
}
function renderWarning(status) {
var el = byId("data-warning");
if (!el) {
return false;
}
var bad = false;
var parts = [];
if (status.incomplete === true) {
bad = true;
}
var dropped = Number(status.dropped_events) || 0;
if (dropped > 0) {
bad = true;
}
if (status.csv_export_ok === false) {
bad = true;
}
if (bad) {
var msg = "DATA";
if (status.last_error) {
msg = msg + status.last_error;
}
parts.push(msg);
}
var gen = String(status.generated_at || "");
var hour = -1;
if (gen.length >= 13) {
hour = Number(gen.slice(11, 13));
}
if (hour >= 19) {
parts.push("laporan hari ini dihapus tengah malam midnight");
}
if (parts.length === 0) {
return false;
}
el.setAttribute("data-state", "warning");
el.textContent = parts.join(" ");
return true;
}
function markLive(published, live) {
var el = byId("data-warning");
if (!el) {
return;
}
if (live) {
el.setAttribute("data-state", "live");
el.textContent = "LIVE ● data terkini";
} else if (published) {
el.setAttribute("data-state", "stale");
el.textContent = "STALE ▲ snapshot lama — menampilkan data terakhir yang valid";
} else {
el.setAttribute("data-state", "loading");
el.textContent = "? menunggu publikasi pertama";
}
}
async function pollOnce() {
if (pending) {
return;
}
pending = true;
try {
var statusRes = await fetch(STATUS_URL);
if (!statusRes.ok) {
throw new Error("status");
}
var status = await statusRes.json();
var summaryRes = await fetch(SUMMARY_URL);
if (!summaryRes.ok) {
throw new Error("summary");
}
var summary = await summaryRes.json();
var seriesRes = await fetch(TIMESERIES_URL);
if (!seriesRes.ok) {
throw new Error("series");
}
var series = await seriesRes.json();
var okSummary = renderSummary(summary);
var okSeries = renderTimeseries(series);
if (!okSummary || !okSeries) {
throw new Error("shape");
}
describeStatus(status);
markLive(status.published, status.live);
renderWarning(status);
lastGood = { status: status, summary: summary, series: series };
} catch (err) {
if (lastGood) {
try {
describeStatus(lastGood.status);
renderSummary(lastGood.summary);
renderTimeseries(lastGood.series);
} catch (inner) {
}
markStale("gagal memuat");
} else {
markStale("gagal memuat");
}
}
pending = false;
timer = setTimeout(pollOnce, POLL_MS);
}
function init() {
var main = byId("main");
if (main) {
var sUrl = main.getAttribute("data-status-url");
if (sUrl) {
STATUS_URL = sUrl;
}
var uUrl = main.getAttribute("data-summary-url");
if (uUrl) {
SUMMARY_URL = uUrl;
}
var tUrl = main.getAttribute("data-timeseries-url");
if (tUrl) {
TIMESERIES_URL = tUrl;
}
}
pollOnce();
}
window.trafficDashboard = { init: init };
})();
