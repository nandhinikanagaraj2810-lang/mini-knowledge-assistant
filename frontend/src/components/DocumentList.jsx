import React from 'react';

function formatDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function DocItem({ doc }) {
  return (
    <div className="doc-item" title={doc.title}>
      <div className="doc-icon">📝</div>
      <div className="doc-info">
        <div className="doc-name">{doc.title}</div>
        <div className="doc-meta">
          {doc.chunk_count} chunk{doc.chunk_count !== 1 ? 's' : ''} ·{' '}
          {(doc.char_count / 1000).toFixed(1)}k chars ·{' '}
          {formatDate(doc.created_at)}
        </div>
      </div>
      <span className="badge badge-green">{doc.status}</span>
    </div>
  );
}

export default function DocumentList({ documents, loading, error, total }) {
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <span className="card-icon">📚</span>
          Knowledge Base
        </div>
        {total > 0 && (
          <span className="count-pill">{total}</span>
        )}
      </div>
      <div className="card-body" style={{ paddingTop: '1rem' }}>
        {loading && (
          <div style={{ padding: '1rem 0' }}>
            {[1, 2, 3].map((i) => (
              <div key={i} style={{ marginBottom: '0.75rem' }}>
                <div className="skeleton skeleton-line" style={{ width: '70%' }} />
                <div className="skeleton skeleton-line" style={{ width: '40%', height: 10 }} />
              </div>
            ))}
          </div>
        )}
        {error && (
          <div className="alert alert-error">
            <span>⚠</span><span>{error}</span>
          </div>
        )}
        {!loading && !error && documents.length === 0 && (
          <div className="empty-state">
            <div className="empty-state-icon">📭</div>
            <p>No documents yet.<br />Upload your first document above.</p>
          </div>
        )}
        {!loading && !error && documents.length > 0 && (
          <div className="doc-list">
            {documents.map((doc) => (
              <DocItem key={doc.id} doc={doc} />
            ))}
          </div>
        )}
        {!loading && total > documents.length && (
          <p style={{ textAlign: 'center', fontSize: '0.75rem', marginTop: '0.75rem', color: 'var(--text-muted)' }}>
            Showing {documents.length} of {total}
          </p>
        )}
      </div>
    </div>
  );
}
