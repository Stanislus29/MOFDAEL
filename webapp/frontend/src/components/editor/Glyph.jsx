/** SVG glyph for each palette component (minimalist, navy line style). */

const base = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

export function Glyph({ type, size = 34 }) {
  const s = size;
  const c = s / 2;
  const r = s * 0.36;

  switch (type) {
    case "source":
      return (
        <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
          <circle cx={c} cy={c} r={r} {...base} />
          <path
            d={`M ${c - r * 0.55} ${c} q ${r * 0.28} -${r * 0.5} ${r * 0.55} 0 q ${r * 0.28} ${r * 0.5} ${r * 0.55} 0`}
            {...base}
          />
        </svg>
      );
    case "substation":
      return (
        <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
          <path
            d={`M ${c - r} ${c + r * 0.75} L ${c - r} ${c} L ${c} ${c - r * 0.9} L ${c + r} ${c} L ${c + r} ${c + r * 0.75} Z`}
            {...base}
          />
          <path d={`M ${c - r * 0.4} ${c + r * 0.75} v -${r * 0.5} M ${c + r * 0.4} ${c + r * 0.75} v -${r * 0.5}`} {...base} />
        </svg>
      );
    case "busbar":
      return (
        <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
          <path d={`M ${s * 0.15} ${c} H ${s * 0.85}`} stroke="currentColor" strokeWidth={3} strokeLinecap="round" />
          <path d={`M ${c} ${s * 0.3} v ${s * 0.12} M ${c} ${s * 0.58} v ${s * 0.12}`} {...base} />
        </svg>
      );
    case "transformer":
      return (
        <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
          <circle cx={c - r * 0.45} cy={c} r={r * 0.72} {...base} />
          <circle cx={c + r * 0.45} cy={c} r={r * 0.72} {...base} />
        </svg>
      );
    case "relay":
      return (
        <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
          <rect x={c - r * 0.8} y={c - r * 0.8} width={r * 1.6} height={r * 1.6} rx={2.5} {...base} />
          <path d={`M ${c - r * 0.4} ${c + r * 0.4} L ${c + r * 0.4} ${c - r * 0.4}`} {...base} />
        </svg>
      );
    case "ctpt":
      return (
        <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
          <circle cx={c} cy={c} r={r * 0.8} {...base} />
          <circle cx={c} cy={c} r={2.2} fill="currentColor" stroke="none" />
        </svg>
      );
    case "load":
      return (
        <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
          <path d={`M ${c} ${c - r} L ${c + r * 0.85} ${c + r * 0.7} L ${c - r * 0.85} ${c + r * 0.7} Z`} {...base} />
        </svg>
      );
    case "junction":
      return (
        <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
          <circle cx={c} cy={c} r={r * 0.45} fill="currentColor" stroke="none" />
          <circle cx={c} cy={c} r={r * 0.8} {...base} />
        </svg>
      );
    default:
      return null;
  }
}
