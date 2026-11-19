import { useRef, useState, useEffect } from "react";
import { NODE_RADIUS } from "../../state/network.js";
import { Glyph } from "../editor/Glyph.jsx";

/** Read-only schematic for the Line Monitor: pan/zoom, nodes, edges,
 * fault heat bands positioned along each mapped segment, relay trip
 * markers. Zone bands use the --fault-1..6 severity ramp; suspected
 * false positives render as hollow dashed bands. */

const WORLD = 4000;
const GRID = 40;
const BAND_OFFSET = 9;
const BAND_HEIGHT = 7;

export default function Schematic({ doc, zonesByEdge, onEdgeClick, selectedEdgeId }) {
  const svgRef = useRef(null);
  const [view, setView] = useState({ x: 0, y: 0, k: 1 });
  const drag = useRef(null);

  const onWheel = (e) => {
    e.preventDefault();
    const rect = svgRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    const k = Math.min(3, Math.max(0.3, view.k * factor));
    setView({
      x: mx - ((mx - view.x) * k) / view.k,
      y: my - ((my - view.y) * k) / view.k,
      k,
    });
  };

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view]);

  const onPointerDown = (e) => {
    if (e.button !== 0 || !e.target.dataset.bg) return;
    drag.current = {
      kind: "pan",
      sx: e.clientX,
      sy: e.clientY,
      ox: view.x,
      oy: view.y,
    };
    svgRef.current.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e) => {
    const d = drag.current;
    if (!d) return;
    setView((v) => ({
      ...v,
      x: d.ox + (e.clientX - d.sx),
      y: d.oy + (e.clientY - d.sy),
    }));
  };

  const onPointerUp = () => {
    drag.current = null;
  };

  const byId = Object.fromEntries(doc.nodes.map((n) => [n.id, n]));

  // relay trip markers: a relay trips when an adjacent mapped segment
  // shows a relay flag fraction >= 0.5
  const trippedRelays = new Set();
  for (const edge of doc.edges) {
    const z = zonesByEdge?.[edge.id];
    if (z && (z.per_model?.relay?.flag_fraction ?? 0) >= 0.5) {
      trippedRelays.add(edge.from);
      trippedRelays.add(edge.to);
    }
  }

  return (
    <div
      style={{
        flex: 1,
        position: "relative",
        border: "1px solid var(--hairline)",
        borderRadius: 8,
        overflow: "hidden",
        background: "#fff",
        minHeight: 420,
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 10,
          right: 12,
          fontSize: 11,
          color: "var(--ink-secondary)",
          zIndex: 2,
          background: "rgba(247,246,242,0.9)",
          padding: "2px 8px",
          borderRadius: 999,
        }}
      >
        {Math.round(view.k * 100)}% · drag to pan
      </div>

      <svg
        ref={svgRef}
        data-schematic="1"
        width="100%"
        height="100%"
        style={{ display: "block" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <defs>
          <pattern id="mgrid" width={GRID} height={GRID} patternUnits="userSpaceOnUse">
            <path d={`M ${GRID} 0 L 0 0 0 ${GRID}`} fill="none" stroke="var(--wash)" strokeWidth="1" />
          </pattern>
        </defs>

        <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}>
          <rect data-bg="1" x={-WORLD / 2} y={-WORLD / 2} width={WORLD} height={WORLD} fill="url(#mgrid)" />

          {/* edges + heat zones */}
          {doc.edges.map((edge) => {
            const a = byId[edge.from];
            const b = byId[edge.to];
            if (!a || !b) return null;
            const zones = zonesByEdge?.[edge.id]?.zones ?? [];
            const selected = selectedEdgeId === edge.id;
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const len = Math.hypot(dx, dy) || 1;
            // unit perpendicular (above the line)
            const px = -dy / len;
            const py = dx / len;
            const sign = py > 0 || (py === 0 && px < 0) ? -1 : 1;

            return (
              <g key={edge.id}>
                <line
                  x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                  stroke={selected ? "var(--deep-blue)" : "var(--ink)"}
                  strokeWidth={selected ? 2.6 : 1.5}
                />
                {zones.map((z, i) => {
                  if (z.loc_min_km == null || z.loc_max_km == null) return null;
                  const L = edge.props?.length_km;
                  if (!L) return null;
                  const f0 = Math.max(0, Math.min(1, z.loc_min_km / L));
                  const f1 = Math.max(0, Math.min(1, z.loc_max_km / L));
                  // minimum visible width ~4% of the segment
                  const fMid = (f0 + f1) / 2;
                  const fHalf = Math.max((f1 - f0) / 2, 0.02);
                  const x0 = a.x + dx * (fMid - fHalf);
                  const y0 = a.y + dy * (fMid - fHalf);
                  const x1 = a.x + dx * (fMid + fHalf);
                  const y1 = a.y + dy * (fMid + fHalf);
                  const off = BAND_OFFSET + i * (BAND_HEIGHT + 3);
                  const cx = (x0 + x1) / 2 + sign * px * off;
                  const cy = (y0 + y1) / 2 + sign * py * off;
                  const w = Math.hypot(x1 - x0, y1 - y0);
                  const angle = (Math.atan2(y1 - y0, x1 - x0) * 180) / Math.PI;
                  const color = `var(--fault-${rampIndex(z.flag_fraction)})`;
                  const fp = z.suspected_false_positive;
                  const pulse = !fp && z.flag_fraction >= 0.75;
                  return (
                    <rect
                      key={i}
                      className={pulse ? "zone-pulse" : undefined}
                      x={-w / 2}
                      y={-BAND_HEIGHT / 2}
                      width={w}
                      height={BAND_HEIGHT}
                      rx={BAND_HEIGHT / 2}
                      transform={`translate(${cx},${cy}) rotate(${angle})`}
                      fill={fp ? "none" : color}
                      stroke={fp ? color : "none"}
                      strokeDasharray={fp ? "3 2" : undefined}
                      strokeWidth={fp ? 1.4 : undefined}
                      opacity={fp ? 0.9 : 0.55 + 0.45 * z.flag_fraction}
                    />
                  );
                })}
                {/* hit line + labels */}
                <g style={{ cursor: "pointer" }} onClick={(e) => { e.stopPropagation(); onEdgeClick?.(edge); }}>
                  <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="transparent" strokeWidth="14" />
                  {edge.props?.label && (
                    <text
                      x={(a.x + b.x) / 2}
                      y={(a.y + b.y) / 2 - 18}
                      textAnchor="middle"
                      fontSize="10.5"
                      fontWeight="600"
                      fill="var(--navy)"
                      style={{ userSelect: "none" }}
                    >
                      {edge.props.label}
                    </text>
                  )}
                  {edge.props?.length_km != null && (
                    <text
                      x={(a.x + b.x) / 2}
                      y={(a.y + b.y) / 2 - 6}
                      textAnchor="middle"
                      fontSize="10"
                      fill="var(--ink-secondary)"
                      style={{ userSelect: "none" }}
                    >
                      {Number(edge.props.length_km).toFixed(0)} km
                    </text>
                  )}
                </g>
              </g>
            );
          })}

          {/* nodes */}
          {doc.nodes.map((node) => {
            const selected = selectedEdgeId && (node.id === doc.edges.find((e) => e.id === selectedEdgeId)?.from || node.id === doc.edges.find((e) => e.id === selectedEdgeId)?.to);
            const tripped = trippedRelays.has(node.id);
            return (
              <g key={node.id} transform={`translate(${node.x},${node.y})`}>
                {selected && (
                  <circle r={NODE_RADIUS + 5} fill="none" stroke="var(--deep-blue)" strokeWidth="1.6" />
                )}
                <circle r={NODE_RADIUS} fill="#fff" stroke="var(--ink)" strokeWidth="1.4" />
                <g transform="translate(-15,-15)" color="var(--navy)">
                  <Glyph type={node.type} size={30} />
                </g>
                {tripped && (
                  <circle
                    r={5}
                    cx={NODE_RADIUS * 0.85}
                    cy={-NODE_RADIUS * 0.85}
                    fill="var(--fault-5)"
                    stroke="#fff"
                    strokeWidth="1.2"
                  />
                )}
                {node.label && (
                  <text y={NODE_RADIUS + 14} textAnchor="middle" fontSize="11" fill="var(--ink)" style={{ userSelect: "none" }}>
                    {node.label}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}

/** Map a consensus fraction (0..1) to the 6-step fault ramp. */
function rampIndex(flagFraction) {
  return Math.max(1, Math.min(6, Math.ceil(flagFraction * 6)));
}
