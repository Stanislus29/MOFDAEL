import { NODE_TYPES } from "../../state/network.js";
import { Glyph } from "./Glyph.jsx";
import { TOOL_SELECT, TOOL_CONNECT } from "../../state/network.js";

export default function Palette({ tool, onTool, onPlace }) {
  return (
    <div style={{ display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap" }}>
      <div>
        <div className="section-title">Tools</div>
        <div style={{ display: "flex", gap: 6 }}>
          <ToolButton active={tool === TOOL_SELECT} onClick={() => onTool(TOOL_SELECT)}>
            Select / move
          </ToolButton>
          <ToolButton active={tool === TOOL_CONNECT} onClick={() => onTool(TOOL_CONNECT)}>
            Draw line
          </ToolButton>
        </div>
      </div>

      <div>
        <div className="section-title">Components — click, then place on canvas</div>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {Object.entries(NODE_TYPES).map(([type, meta]) => (
            <button
              key={type}
              className="palette-item"
              style={paletteStyle}
              onClick={() => onPlace(type)}
              title={`Place ${meta.label}`}
            >
              <Glyph type={type} size={30} />
              <span style={{ fontSize: 11 }}>{meta.label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function ToolButton({ active, children, ...rest }) {
  return (
    <button
      style={{
        ...toolStyle,
        border: `1px solid ${active ? "var(--navy)" : "var(--hairline)"}`,
        ...(active ? { background: "var(--navy)", color: "#fff" } : {}),
      }}
      {...rest}
    >
      {children}
    </button>
  );
}

const toolStyle = {
  padding: "6px 12px",
  borderRadius: 6,
  border: "1px solid var(--hairline)",
  background: "#fff",
  color: "var(--ink)",
  font: "inherit",
  fontSize: 12.5,
  cursor: "pointer",
};

const paletteStyle = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: 2,
  width: 64,
  padding: "8px 4px 6px",
  borderRadius: 6,
  border: "1px solid var(--hairline)",
  background: "#fff",
  color: "var(--navy)",
  cursor: "pointer",
  font: "inherit",
};
