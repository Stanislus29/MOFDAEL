/** Small shared UI primitives. */

export function Button({
  variant = "primary",
  children,
  className = "",
  ...rest
}) {
  return (
    <button className={`btn btn-${variant} ${className}`} {...rest}>
      {children}
    </button>
  );
}

export function SectionTitle({ children }) {
  return <div className="section-title">{children}</div>;
}

export function Hairline() {
  return <hr className="hairline" />;
}

export function Chip({ accent = false, children }) {
  return <span className={`chip${accent ? " chip-accent" : ""}`}>{children}</span>;
}
