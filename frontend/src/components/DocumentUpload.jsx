import React, { useState, useRef } from 'react';
import { uploadDocument } from '../services/api';

export default function DocumentUpload({ onSuccess }) {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [status, setStatus] = useState(null); // null | 'loading' | 'success' | 'error'
  const [message, setMessage] = useState('');
  const titleRef = useRef(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!title.trim() || !content.trim()) {
      setStatus('error');
      setMessage('Both title and content are required.');
      return;
    }
    setStatus('loading');
    setMessage('');
    try {
      const doc = await uploadDocument(title.trim(), content.trim());
      setStatus('success');
      setMessage(`"${doc.title}" indexed successfully — ${doc.chunk_count} chunk(s) created.`);
      setTitle('');
      setContent('');
      if (onSuccess) onSuccess(doc);
      titleRef.current?.focus();
    } catch (err) {
      setStatus('error');
      setMessage(err.message || 'Upload failed. Please try again.');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) handleSubmit(e);
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <span className="card-icon">📄</span>
          Add Document
        </div>
      </div>
      <div className="card-body">
        {status === 'success' && (
          <div className="alert alert-success" role="alert">
            <span>✓</span>
            <span>{message}</span>
          </div>
        )}
        {status === 'error' && (
          <div className="alert alert-error" role="alert">
            <span>✕</span>
            <span>{message}</span>
          </div>
        )}
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="doc-title">Title</label>
            <input
              id="doc-title"
              ref={titleRef}
              type="text"
              placeholder="e.g. Refund Policy"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              disabled={status === 'loading'}
              maxLength={512}
              autoComplete="off"
            />
          </div>
          <div className="form-group">
            <label htmlFor="doc-content">Content</label>
            <textarea
              id="doc-content"
              placeholder="Paste your document content here…"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              disabled={status === 'loading'}
              onKeyDown={handleKeyDown}
              style={{ minHeight: 120 }}
            />
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              Ctrl+Enter to submit · Max 100,000 characters
            </div>
          </div>
          <button
            id="upload-btn"
            type="submit"
            className="btn btn-primary btn-full"
            disabled={status === 'loading'}
          >
            {status === 'loading' ? (
              <><span className="spinner" />Indexing…</>
            ) : (
              <><span>⬆</span> Upload & Index</>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
