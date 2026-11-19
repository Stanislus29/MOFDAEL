import { useReducer, useEffect, useRef, useState, useCallback } from "react";
import Canvas from "../components/editor/Canvas.jsx";
import Palette from "../components/editor/Palette.jsx";
import InspectorPanel from "../components/editor/InspectorPanel.jsx";
import {
  reducer,
  initialState,
  serialize,
  EMPTY_DOC,
  TOOL_SELECT,
  TOOL_CONNECT,
} from "../state/network.js";
import { api } from "../api/client.js";
import { Button, Chip, SectionTitle } from "../components/ui.jsx";

export default function StudioView() {
  const [state, dispatch] = useReducer(reducer, undefined, initialState);
  const [placeType, setPlaceType] = useState(null);
  const [tool, setTool] = useState(TOOL_SELECT);
  const [networks, setNetworks] = useState([]);
  const [status, setStatus] = useState("");
  const importRef = useRef(null);

  const refreshNetworks = useCallback(() => {
    api.listNetworks().then((r) => setNetworks(r.networks)).catch(() => setNetworks([]));
  }, []);

  useEffect(refreshNetworks, [refreshNetworks]);

  // -------------------------------------------------- interactions

  const handleCanvasClick = (x, y) => {
    if (placeType) {
      dispatch({ type: "add_node", nodeType: placeType, x, y });
      setPlaceType(null);
    }
  };

  const handleNodeClick = (node) => {
    if (tool === TOOL_CONNECT) {
      if (state.pendingFrom && state.pendingFrom !== node.id) {
        dispatch({ type: "connect", from: state.pendingFrom, to: node.id });
      } else {
        dispatch({ type: "pending_from", nodeId: node.id });
      }
    } else {
      dispatch({ type: "select", selection: { kind: "node", id: node.id } });
    }
  };

  const handleEdgeClick = (edge) =>
    dispatch({ type: "select", selection: { kind: "edge", id: edge.id } });

  const handleBackgroundClick = () => {
    if (!placeType) dispatch({ type: "select", selection: null });
  };

  // Delete key removes the selection
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Delete" || e.key === "Backspace") {
        const tag = document.activeElement?.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        if (state.selection) dispatch({ type: "delete_selection" });
      }
      if (e.key === "Escape") setPlaceType(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state.selection]);

  // -------------------------------------------------- persistence

  const save = async () => {
    setStatus("Saving…");
    try {
      const doc = serialize(state);
      const res = await api.saveNetwork({
        network_id: state.networkId,
        name: state.name,
        doc,
      });
      dispatch({ type: "mark_saved", networkId: res.network_id });
      refreshNetworks();
      setStatus(`Saved · ${new Date().toLocaleTimeString()}`);
    } catch (e) {
      setStatus(`Save failed — ${e.message}`);
    }
  };

  const load = async (networkId) => {
    try {
      const rec = await api.getNetwork(networkId);
      dispatch({
        type: "load",
        doc: rec.doc,
        name: rec.name,
        networkId: rec.network_id,
      });
      setStatus(`Loaded “${rec.name}”`);
    } catch (e) {
      setStatus(`Load failed — ${e.message}`);
    }
  };

  const exportJson = () => {
    const blob = new Blob([JSON.stringify(serialize(state), null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${state.name.replace(/\s+/g, "_")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const importJson = async (file) => {
    try {
      const text = await file.text();
      const doc = JSON.parse(text);
      dispatch({
        type: "load",
        doc,
        name: doc.name || file.name.replace(/\.json$/, ""),
        networkId: null,
      });
      setStatus(`Imported ${file.name}`);
    } catch (e) {
      setStatus(`Import failed — ${e.message}`);
    }
  };

  const newNetwork = () => {
    dispatch({ type: "load", doc: EMPTY_DOC, name: "Untitled network", networkId: null });
    setStatus("New network");
  };

  const nodeCount = state.doc.nodes.length;
  const edgeCount = state.doc.edges.length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18, minHeight: "calc(100% - 56px)" }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <input
          value={state.name}
          onChange={(e) => dispatch({ type: "rename", name: e.target.value })}
          style={{
            padding: "7px 12px",
            borderRadius: 6,
            border: "1px solid var(--hairline)",
            font: "inherit",
            fontWeight: 600,
            color: "var(--ink)",
            width: 240,
            background: "#fff",
          }}
        />
        <Chip accent={state.dirty}>{state.dirty ? "unsaved changes" : "saved"}</Chip>
        <Chip>
          {nodeCount} component{nodeCount === 1 ? "" : "s"} · {edgeCount} segment{edgeCount === 1 ? "" : "s"}
        </Chip>

        <span style={{ flex: 1 }} />

        <Button variant="ghost" onClick={newNetwork}>New</Button>
        <Button variant="ghost" disabled={!state.networkId && !state.dirty} onClick={save}>Save</Button>
        <Button variant="ghost" onClick={exportJson}>Export JSON</Button>
        <Button variant="ghost" onClick={() => importRef.current?.click()}>Import JSON</Button>
        <input
          ref={importRef}
          type="file"
          accept=".json"
          hidden
          onChange={(e) => e.target.files?.[0] && importJson(e.target.files[0])}
        />
      </div>

      <div>
        <Palette
          tool={tool}
          onTool={(t) => {
            setTool(t);
            setPlaceType(null);
            dispatch({ type: "select", selection: null });
          }}
          onPlace={(type) => {
            setTool(TOOL_SELECT);
            setPlaceType(type);
          }}
        />
        {/* reserved status line: constant height so the canvas never
            jumps when hints appear/disappear */}
        <div style={{ marginTop: 10, minHeight: 22, fontSize: 12.5, color: "var(--deep-blue)" }}>
          {placeType
            ? `Click on the canvas to place the ${placeType}. Esc to cancel.`
            : tool === TOOL_CONNECT
              ? "Click a start component, then an end component to draw a line segment."
              : status}
        </div>
      </div>

      <div style={{ display: "flex", flex: 1, gap: 0 }}>
        <Canvas
          doc={state.doc}
          selection={state.selection}
          pendingFrom={state.pendingFrom}
          tool={tool}
          placeType={placeType}
          onCanvasClick={handleCanvasClick}
          onNodeClick={handleNodeClick}
          onNodeDrag={(id, x, y) => dispatch({ type: "move_node", id, x, y })}
          onEdgeClick={handleEdgeClick}
          onBackgroundClick={handleBackgroundClick}
        />
        <InspectorPanel
          doc={state.doc}
          selection={state.selection}
          onNodeChange={(id, patch) => dispatch({ type: "update_node", id, patch })}
          onEdgeChange={(id, patch) => dispatch({ type: "update_edge", id, patch })}
          onDelete={() => dispatch({ type: "delete_selection" })}
        />
      </div>

      {networks.length > 0 && (
        <div>
          <SectionTitle>Saved networks</SectionTitle>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {networks.map((n) => (
              <button
                key={n.network_id}
                className="btn btn-ghost"
                onClick={() => load(n.network_id)}
                title={`Updated ${n.updated_at}`}
              >
                {n.name}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
