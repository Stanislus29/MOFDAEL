/** Chronological-ish event feed: every flagged fault zone across all
 * mapped segments, most severe first. */

import { phasesText } from "./SegmentPanel.jsx";

export default function EventFeed({ doc, zonesByEdge }) {
  const events = [];
  for (const edge of doc.edges) {
    const z = zonesByEdge?.[edge.id];
    if (!z) continue;
    for (const zone of z.zones) {
      events.push({ edge, zone, segment: z.segment });
    }
  }
  events.sort(
    (a, b) =>
      (b.zone.suspected_false_positive ? 0 : b.zone.flag_fraction) -
      (a.zone.suspected_false_positive ? 0 : a.zone.flag_fraction)
  );

  return (
    <div data-event-feed="1" style={{ borderTop: "1px solid var(--hairline)", padding: "12px 18px", maxHeight: 180, overflowY: "auto" }}>
      <div style={{ fontSize: 11.5, color: "var(--ink-secondary)", marginBottom: 8 }}>
        EVENT FEED · {events.length} flagged zone{events.length === 1 ? "" : "s"}
      </div>
      {events.length === 0 ? (
        <div style={{ fontSize: 13, color: "var(--ink-secondary)" }}>
          Nothing flagged. All monitored segments are quiet.
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {events.map(({ edge, zone, segment }, i) => {
            const fp = zone.suspected_false_positive;
            return (
              <div
                key={i}
                style={{
                  display: "flex",
                  gap: 10,
                  alignItems: "baseline",
                  fontSize: 13,
                  padding: "3px 0",
                  borderBottom: "1px solid var(--wash)",
                }}
              >
                <span
                  style={{
                    width: 9,
                    height: 9,
                    borderRadius: 999,
                    flex: "0 0 9px",
                    alignSelf: "center",
                    background: fp ? "transparent" : `var(--fault-${ramp(zone.flag_fraction)})`,
                    border: `1.4px solid var(--fault-${ramp(zone.flag_fraction)})`,
                  }}
                />
                <span style={{ fontWeight: 600, minWidth: 90 }}>{segment || edge.props?.label || edge.id}</span>
                <span style={{ minWidth: 110, fontVariantNumeric: "tabular-nums" }}>
                  {zone.loc_min_km != null
                    ? zone.loc_min_km === zone.loc_max_km
                      ? `${zone.loc_mean_km.toFixed(1)} km`
                      : `${zone.loc_min_km.toFixed(1)}–${zone.loc_max_km.toFixed(1)} km`
                    : "loc unknown"}
                </span>
                <span style={{ color: "var(--ink-secondary)" }}>{phasesText(zone.phases)}</span>
                <span style={{ marginLeft: "auto", fontVariantNumeric: "tabular-nums", color: fp ? "#b04a3a" : "var(--navy)", fontWeight: 600 }}>
                  {fp
                    ? "suspected false positive"
                    : `${Math.round(zone.flag_fraction * 100)}% of methods flagged`}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ramp(f) {
  return Math.max(1, Math.min(6, Math.ceil(f * 6)));
}
