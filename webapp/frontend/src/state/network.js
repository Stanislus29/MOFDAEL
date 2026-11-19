/** Network document state: nodes (stations, transformers, relays, …)
 * and edges (line segments). Pure reducer + serialization helpers —
 * no rendering concerns here. */

export const NODE_TYPES = {
  source: { label: "Source", terminalCount: 1 },
  substation: { label: "Substation", terminalCount: 4 },
  busbar: { label: "Busbar", terminalCount: 4 },
  transformer: { label: "Transformer", terminalCount: 2 },
  relay: { label: "Relay", terminalCount: 2 },
  ctpt: { label: "CT / PT", terminalCount: 2 },
  load: { label: "Load", terminalCount: 1 },
  junction: { label: "Junction", terminalCount: 4 },
};

export const TOOL_SELECT = "select";
export const TOOL_CONNECT = "connect";

let _seq = 0;
export const uid = (prefix) =>
  `${prefix}_${Date.now().toString(36)}_${(_seq++).toString(36)}`;

export const EMPTY_DOC = { version: 1, nodes: [], edges: [] };

// Default line parameters from the verified physics (Table 3.2).
export const DEFAULT_EDGE_PROPS = {
  length_km: 40,
  r1_per_km: 0.08,
  x1_per_km: 0.39,
  r0_per_km: 0.228,
  x0_per_km: 1.352,
};

export const NODE_RADIUS = 18;

/** Create the initial store state. */
export function initialState() {
  return {
    doc: { ...EMPTY_DOC, nodes: [], edges: [] },
    selection: null, // { kind: 'node'|'edge', id }
    pendingFrom: null, // node id awaiting the second click in connect mode
    name: "Untitled network",
    networkId: null,
    dirty: false,
  };
}

/** Reducer for all editor mutations. */
export function reducer(state, action) {
  switch (action.type) {
    case "load": {
      return {
        ...state,
        doc: { version: 1, nodes: action.doc.nodes || [], edges: action.doc.edges || [] },
        name: action.name ?? state.name,
        networkId: action.networkId ?? state.networkId,
        selection: null,
        pendingFrom: null,
        dirty: false,
      };
    }
    case "rename":
      return { ...state, name: action.name, dirty: true };

    case "add_node": {
      const node = {
        id: uid(action.nodeType),
        type: action.nodeType,
        x: snap(action.x),
        y: snap(action.y),
        label: NODE_TYPES[action.nodeType].label,
        props: {},
      };
      return {
        ...state,
        doc: { ...state.doc, nodes: [...state.doc.nodes, node] },
        selection: { kind: "node", id: node.id },
        dirty: true,
      };
    }

    case "move_node": {
      return {
        ...state,
        doc: {
          ...state.doc,
          nodes: state.doc.nodes.map((n) =>
            n.id === action.id
              ? { ...n, x: snap(action.x), y: snap(action.y) }
              : n
          ),
        },
        dirty: true,
      };
    }

    case "update_node": {
      return {
        ...state,
        doc: {
          ...state.doc,
          nodes: state.doc.nodes.map((n) =>
            n.id === action.id ? { ...n, ...action.patch } : n
          ),
        },
        dirty: true,
      };
    }

    case "update_edge": {
      return {
        ...state,
        doc: {
          ...state.doc,
          edges: state.doc.edges.map((e) =>
            e.id === action.id
              ? { ...e, props: { ...e.props, ...action.patch } }
              : e
          ),
        },
        dirty: true,
      };
    }

    case "connect": {
      const { from, to } = action;
      if (from === to) return state;
      const exists = state.doc.edges.some(
        (e) =>
          (e.from === from && e.to === to) ||
          (e.from === to && e.to === from)
      );
      if (exists) return state;
      const edge = {
        id: uid("edge"),
        from,
        to,
        props: { ...DEFAULT_EDGE_PROPS },
      };
      return {
        ...state,
        doc: { ...state.doc, edges: [...state.doc.edges, edge] },
        pendingFrom: null,
        selection: { kind: "edge", id: edge.id },
        dirty: true,
      };
    }

    case "delete_selection": {
      if (!state.selection) return state;
      const { kind, id } = state.selection;
      const doc =
        kind === "node"
          ? {
              ...state.doc,
              nodes: state.doc.nodes.filter((n) => n.id !== id),
              edges: state.doc.edges.filter(
                (e) => e.from !== id && e.to !== id
              ),
            }
          : {
              ...state.doc,
              edges: state.doc.edges.filter((e) => e.id !== id),
            };
      return {
        ...state,
        doc,
        selection: null,
        pendingFrom: null,
        dirty: true,
      };
    }

    case "select":
      return { ...state, selection: action.selection, pendingFrom: null };

    case "pending_from":
      return { ...state, pendingFrom: action.nodeId, selection: null };

    case "mark_saved":
      return { ...state, dirty: false, networkId: action.networkId ?? state.networkId };

    default:
      return state;
  }
}

function snap(v) {
  return Math.round(v / 10) * 10;
}

/** Serialize a document for persistence / export. */
export function serialize(state) {
  return {
    version: 1,
    name: state.name,
    nodes: state.doc.nodes,
    edges: state.doc.edges,
  };
}
