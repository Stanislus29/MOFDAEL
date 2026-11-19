/** Detail panel for a selected line segment: status, fault zones and
 * per-model verdicts (including the distance relay baseline). */

import { Hairline } from "../ui.jsx";

export default function SegmentPanel({ doc, edge, zonesForEdge, modelName }) {
  if (!edge) {
    return (
      <div style={{ width: 300, flex: "0 0 300px", borderLeft: "1px solid var(--hairline)", padding: 18, color: "var(--ink-secondary)", fontSize: 13 }}>
        Click a line segment to inspect its fault zones and model verdicts.
      </div>
    );
  }

  const z = zonesForEdge?.[edge.id];
  const label = edge.props?.label || edge.id;
  const zones = z?.zones ?? [];

  return (
    <div
      data-segment-panel="1"
      style={{
        width: 300,
        flex: "0 0 300px",
        borderLeft: "1px solid var(--hairline)",
        padding: "18px 18px",
        overflowY: "auto",
      }}
    >
      <div className="section-title">{label}</div>
      <div style={{ fontSize: 12, color: "var(--ink-secondary)", marginBottom: 4 }}>
        {edge.props?.length_km != null && `${Number(edge.props.length_km).toFixed(0)} km · `}
        {z ? `${z.n_rows} rows · ${z.n_flagged} flagged` : "no run data mapped"}
      </div>
      {z && (
        <StatusChip fraction={z.flagged_fraction} />
      )}

      <Hairline />

      <div style={{ fontSize: 11.5, color: "var(--ink-secondary)", marginBottom: 8 }}>
        FAULT ZONES
      </div>
      {!z || zones.length === 0 ? (
        <div style={{ fontSize: 13, color: "var(--ink-secondary)" }}>
          No flagged fault events on this segment.
        </div>
      ) : (
        zones.map((zone, i) => (
          <div
            key={i}
            style={{
              border: "1px solid var(--hairline)",
              borderRadius: 8,
              padding: "10px 12px",
              marginBottom: 8,
              background: zone.suspected_false_positive ? "transparent" : "#fff",
              borderStyle: zone.suspected_false_positive ? "dashed" : "solid",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <span style={{ fontWeight: 600, fontSize: 13.5 }}>
                {zone.loc_min_km != null
                  ? zone.loc_min_km === zone.loc_max_km
                    ? `${zone.loc_mean_km.toFixed(1)} km`
                    : `${zone.loc_min_km.toFixed(1)} – ${zone.loc_max_km.toFixed(1)} km`
                  : "location unknown"}
              </span>
              <span className="mono" style={{ fontSize: 12, color: "var(--navy)" }}>
                {Math.round(zone.flag_fraction * 100)}%
              </span>
            </div>
            <div style={{ fontSize: 12, color: "var(--ink-secondary)", marginTop: 3 }}>
              {zone.n_rows} rows · {phasesText(zone.phases)}
              {zone.suspected_false_positive && (
                <span style={{ color: "#b04a3a", fontWeight: 600 }}> · suspected false positive</span>
              )}
            </div>
            {zone.position_source === "loc_pred" && (
              <div style={{ fontSize: 11, color: "var(--ink-secondary)", marginTop: 3 }}>
                positioned by model-estimated location
              </div>
            )}
          </div>
        ))
      )}

      {z && (
        <>
          <Hairline />
          <div style={{ fontSize: 11.5, color: "var(--ink-secondary)", marginBottom: 8 }}>
            MODEL VERDICTS
          </div>
          <table className="data-table" style={{ width: "100%" }}>
            <thead>
              <tr>
                <th>Method</th>
                <th>Flag rate</th>
                <th>Conf.</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(z.per_model).map(([id, m]) => (
                <tr key={id}>
                  <td style={{ fontWeight: 600 }}>{modelName ? modelName(id) : id}</td>
                  <td className="mono">{Math.round((m.flag_fraction ?? 0) * 100)}%</td>
                  <td className="mono">{m.mean_confidence != null ? m.mean_confidence.toFixed(3) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

function StatusChip({ fraction }) {
  const state =
    fraction === 0 ? "healthy" : fraction >= 0.5 ? "fault activity" : "scattered flags";
  const accent = fraction === 0 ? "var(--ink-secondary)" : fraction >= 0.5 ? "var(--fault-5)" : "var(--navy)";
  return (
    <div style={{ marginTop: 8, marginBottom: 4, fontSize: 13 }}>
      <span
        style={{
          display: "inline-block",
          padding: "3px 10px",
          borderRadius: 999,
          border: `1px solid ${accent}`,
          color: accent,
          fontWeight: 600,
          fontSize: 12,
        }}
      >
        {state} · {Math.round(fraction * 100)}%
      </span>
    </div>
  );
}

export function phasesText(phases) {
  if (!phases) return "";
  const on = Object.entries(phases)
    .filter(([, v]) => v > 0)
    .map(([k]) => k.toUpperCase());
  return on.length ? `phases ${on.join(", ")}` : "no phase consensus";
}
