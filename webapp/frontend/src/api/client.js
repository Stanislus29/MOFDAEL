/** Thin fetch wrapper around the platform API. */

const BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options);
  const text = await res.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) {
    const message =
      (body && (body.error || body.detail)) ||
      `Request failed (${res.status})`;
    throw new Error(message);
  }
  return body;
}

export const api = {
  health: () => request("/health"),

  listModels: () => request("/models"),

  uploadDataset: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/datasets", { method: "POST", body: form });
  },

  listDatasets: () => request("/datasets"),

  datasetDetail: (id) => request(`/datasets/${id}`),

  datasetPreview: (id, rows = 20) =>
    request(`/datasets/${id}/preview?rows=${rows}`),

  startRun: (payload) =>
    request("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  listRuns: () => request("/runs"),

  runDetail: (id) => request(`/runs/${id}`),

  runZones: (id) => request(`/runs/${id}/zones`),

  selfcheck: () => request("/selfcheck", { method: "POST" }),

  // networks
  listNetworks: () => request("/networks"),

  saveNetwork: (payload) =>
    request("/networks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  getNetwork: (id) => request(`/networks/${id}`),

  updateNetwork: (id, payload) =>
    request(`/networks/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  deleteNetwork: (id) => request(`/networks/${id}`, { method: "DELETE" }),
};
