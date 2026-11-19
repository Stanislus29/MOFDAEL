export default function EmptyState({ icon, title, hint, action }) {
  return (
    <div className="empty">
      {icon && <div className="empty-icon">{icon}</div>}
      <div className="empty-title">{title}</div>
      <div className="empty-hint">{hint}</div>
      {action}
    </div>
  );
}
