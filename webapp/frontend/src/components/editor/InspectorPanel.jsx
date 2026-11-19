import { NODE_TYPES } from "../../state/network.js";

/** Right-hand inspector: edit the selected node or line segment. */

export default function InspectorPanel({ doc, selection, onNodeChange, onEdgeChange, onDelete }) {
  const node = selection?.kind === "node" ? doc.nodes.find((n) => n.id === selection.id) : null;
  const edge = selection?.kind === "edge" ? doc.edges.find((e) => e.id === selection.id) : null;

  return (
    <div data-inspector="1" style={{ width: 280, flex: "0 0 280px", borderLeft: "1px solid var(--hairline)", padding: "18px 18px" }}>
      {!node && !edge && (
        <div style={{ color: "var(--ink-secondary)", fontSize: 13 }}>
          Select a component or line segment to edit its properties.
        </div>
      )}

      {node && (
        <>
          <div className="section-title">{NODE_TYPES[node.type].label}</div>
          <Field label="Label">
            <input
              style={inputStyle}
              value={node.label}
              onChange={(e) => onNodeChange(node.id, { label: e.target.value })}
            />
          </Field>
          <div style={{ marginTop: 14 }}>
            <button className="btn btn-ghost" style={{ color: "#b04a3a" }} onClick={onDelete}>
              Delete component
            </button>
          </div>
        </>
      )}

      {edge && (
        <>
          <div className="section-title">Line segment</div>
          <Field label="Segment name (matches CSV `segment` column)">
            <input
              style={inputStyle}
              placeholder="e.g. Feeder A"
              value={edge.props.label ?? ""}
              onChange={(e) => onEdgeChange(edge.id, { label: e.target.value })}
            />
          </Field>
          <Field label="Length (km)">
            <input
              type="number"
              min="0"
              step="1"
              style={inputStyle}
              value={edge.props.length_km}
              onChange={(e) => onEdgeChange(edge.id, { length_km: Number(e.target.value) })}
            />
          </Field>
          <Field label="Z₁ R (Ω/km)">
            <input
              type="number"
              step="0.001"
              style={inputStyle}
              value={edge.props.r1_per_km}
              onChange={(e) => onEdgeChange(edge.id, { r1_per_km: Number(e.target.value) })}
            />
          </Field>
          <Field label="Z₁ X (Ω/km)">
            <input
              type="number"
              step="0.001"
              style={inputStyle}
              value={edge.props.x1_per_km}
              onChange={(e) => onEdgeChange(edge.id, { x1_per_km: Number(e.target.value) })}
            />
          </Field>
          <Field label="Z₀ R (Ω/km)">
            <input
              type="number"
              step="0.001"
              style={inputStyle}
              value={edge.props.r0_per_km}
              onChange={(e) => onEdgeChange(edge.id, { r0_per_km: Number(e.target.value) })}
            />
          </Field>
          <Field label="Z₀ X (Ω/km)">
            <input
              type="number"
              step="0.001"
              style={inputStyle}
              value={edge.props.x0_per_km}
              onChange={(e) => onEdgeChange(edge.id, { x0_per_km: Number(e.target.value) })}
            />
          </Field>
          <div style={{ marginTop: 14 }}>
            <button className="btn btn-ghost" style={{ color: "#b04a3a" }} onClick={onDelete}>
              Delete segment
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label style={{ display: "block", marginTop: 10 }}>
      <div style={{ fontSize: 11.5, color: "var(--ink-secondary)", marginBottom: 4 }}>{label}</div>
      {children}
    </label>
  );
}

const inputStyle = {
  width: "100%",
  padding: "7px 10px",
  borderRadius: 6,
  border: "1px solid var(--hairline)",
  font: "inherit",
  fontSize: 13,
  color: "var(--ink)",
  background: "#fff",
};
