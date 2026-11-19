import { useRef, useState, useCallback, useEffect } from "react";
import { NODE_TYPES, NODE_RADIUS } from "../../state/network.js";
import { Glyph } from "./Glyph.jsx";

/** SVG network canvas: pan (drag background), zoom (wheel),
 * place/drag nodes, click-connect nodes in connect mode. */

const WORLD = 4000; // logical canvas size
const GRID = 40;

export default function Canvas({
  doc,
  selection,
  pendingFrom,
  tool,
  placeType,
  onCanvasClick,
  onNodeClick,
  onNodeDrag,
  onEdgeClick,
  onBackgroundClick,
}) {
  const svgRef = useRef(null);
  const [view, setView] = useState({ x: 0, y: 0, k: 1 });
  const drag = useRef(null); // { kind:'pan'|'node', ... }
  const [hoverNode, setHoverNode] = useState(null);

  // -------------------------------------------------- coordinates

  const toWorld = useCallback(
    (clientX, clientY) => {
      const rect = svgRef.current.getBoundingClientRect();
      return {
        x: (clientX - rect.left - view.x) / view.k,
        y: (clientY - rect.top - view.y) / view.k,
      };
    },
    [view]
  );

  const onWheel = (e) => {
    e.preventDefault();
    const rect = svgRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    const k = Math.min(3, Math.max(0.3, view.k * factor));
    const x = mx - ((mx - view.x) * k) / view.k;
    const y = my - ((my - view.y) * k) / view.k;
    setView({ x, y, k });
  };

  // -------------------------------------------------- pointer handlers

  const onPointerDown = (e) => {
    if (e.button !== 0) return;
    if (e.target.dataset.bg) {
      const p = toWorld(e.clientX, e.clientY);
      // Placement happens on press; pan only when not placing.
      if (placeType) {
        onCanvasClick?.(p.x, p.y);
        return;
      }
      drag.current = { kind: "pan", sx: e.clientX, sy: e.clientY, ox: view.x, oy: view.y };
      svgRef.current.setPointerCapture(e.pointerId);
      onBackgroundClick?.();
    }
  };

  const onNodePointerDown = (e, node) => {
    e.stopPropagation();
    if (tool === "connect") {
      onNodeClick?.(node, e);
      return;
    }
    const p = toWorld(e.clientX, e.clientY);
    drag.current = {
      kind: "node",
      id: node.id,
      dx: node.x - p.x,
      dy: node.y - p.y,
      moved: false,
      pointerId: e.pointerId,
    };
    svgRef.current.setPointerCapture(e.pointerId);
    onNodeClick?.(node, e);
  };

    const onPointerMove = (e) => {
    const d = drag.current;
    if (!d) return;
    if (d.kind === "pan") {
      setView((v) => ({ ...v, x: d.ox + (e.clientX - d.sx), y: d.oy + (e.clientY - d.sy) }));
    } else if (d.kind === "node") {
      const p = toWorld(e.clientX, e.clientY);
      d.moved = true;
      onNodeDrag?.(d.id, p.x + d.dx, p.y + d.dy);
    }
  };

  const onPointerUp = () => {
    drag.current = null;
  };

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view]);

  const byId = Object.fromEntries(doc.nodes.map((n) => [n.id, n]));

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
        {Math.round(view.k * 100)}% · drag background to pan
      </div>

      <svg
        ref={svgRef}
        data-canvas="1"
        width="100%"
        height="100%"
        style={{ display: "block", cursor: placeType ? "copy" : "default" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <defs>
          <pattern id="grid" width={GRID} height={GRID} patternUnits="userSpaceOnUse">
            <path d={`M ${GRID} 0 L 0 0 0 ${GRID}`} fill="none" stroke="var(--wash)" strokeWidth="1" />
          </pattern>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--hairline)" />
          </marker>
        </defs>

        <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}>
          {/* world background (receives pan/placement clicks) */}
          <rect
            data-bg="1"
            x={-WORLD / 2}
            y={-WORLD / 2}
            width={WORLD}
            height={WORLD}
            fill="url(#grid)"
          />

          {/* edges */}
          {doc.edges.map((edge) => {
            const a = byId[edge.from];
            const b = byId[edge.to];
            if (!a || !b) return null;
            const selected = selection?.kind === "edge" && selection.id === edge.id;
            return (
              <g key={edge.id} onClick={(e) => { e.stopPropagation(); onEdgeClick?.(edge); }}>
                <line
                  x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                  stroke={selected ? "var(--deep-blue)" : "var(--ink)"}
                  strokeWidth={selected ? 2.4 : 1.4}
                  markerEnd="url(#arrow)"
                  markerStart="url(#arrow)"
                />
                {edge.props?.label && (
                  <text
                    x={(a.x + b.x) / 2}
                    y={(a.y + b.y) / 2 - (edge.props?.length_km != null ? 18 : 6)}
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
                    fontSize="10.5"
                    fill="var(--ink-secondary)"
                    style={{ userSelect: "none" }}
                  >
                    {Number(edge.props.length_km).toFixed(0)} km
                  </text>
                )}
                {/* fat invisible hit line */}
                <line
                  x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                  stroke="transparent" strokeWidth="12"
                />
              </g>
            );
          })}

          {/* nodes */}
          {doc.nodes.map((node) => {
            const selected = selection?.kind === "node" && selection.id === node.id;
            const pending = pendingFrom === node.id;
            const hovered = hoverNode === node.id;
            return (
              <g
                key={node.id}
                transform={`translate(${node.x},${node.y})`}
                style={{ cursor: tool === "connect" ? "crosshair" : "move" }}
                onPointerDown={(e) => onNodePointerDown(e, node)}
                onPointerEnter={() => setHoverNode(node.id)}
                onPointerLeave={() => setHoverNode(null)}
                onClick={(e) => e.stopPropagation()}
              >
                {(selected || pending) && (
                  <circle r={NODE_RADIUS + 5} fill="none" stroke="var(--deep-blue)" strokeWidth="1.6" strokeDasharray={pending ? "4 3" : undefined} />
                )}
                <circle
                  r={NODE_RADIUS}
                  fill="#fff"
                  stroke={selected || pending ? "var(--deep-blue)" : hovered ? "var(--navy)" : "var(--ink)"}
                  strokeWidth="1.4"
                />
                <g transform="translate(-15,-15)" color="var(--navy)">
                  <Glyph type={node.type} size={30} />
                </g>
                <text y={NODE_RADIUS + 14} textAnchor="middle" fontSize="11" fill="var(--ink)" style={{ userSelect: "none" }}>
                  {node.label}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}
