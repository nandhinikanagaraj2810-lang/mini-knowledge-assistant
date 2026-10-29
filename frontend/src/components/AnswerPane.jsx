import React from 'react';
import { dedupeSources } from '../utils/dedupeSources';

function SourceCard({ source }) {
  const pct = Math.round(source.relevance_score * 100);
  return (
    <div className="source-card">
      <div className="source-header">
        <div className="source-title" title={source.title}>📄 {source.title}</div>
        <div className="relevance-bar-wrap">
          <span className="relevance-score">{pct}%</span>
          <div className="relevance-bar" title={`Relevance: ${pct}%`}>
            <div className="relevance-fill" style={{ width: `${pct}%` }} />
          </div>
        </div>
      </div>
      <p className="source-excerpt">"{source.excerpt}"</p>
    </div>
  );
}

function SkeletonAnswer() {
  return (
    <div style={{ padding: '0.5rem 0' }}>
      <div className="skeleton skeleton-line" style={{ width: '90%' }} />
      <div className="skeleton skeleton-line" style={{ width: '75%' }} />
      <div className="skeleton skeleton-line" style={{ width: '85%' }} />
      <div className="skeleton skeleton-line" />
    </div>
  );
}

export default function AnswerPane({ result, loading, error }) {
  if (!result && !loading && !error) {
    return (
      <div className="empty-state" style={{ minHeight: 200 }}>
        <div className="empty-state-icon">🤖</div>
        <p>Ask a question to get started.<br />Answers are grounded in your uploaded documents.</p>
      </div>
    );
  }

  const sources = result?.sources ? dedupeSources(result.sources) : null;

  return (
    <div className="ask-scroll-area">
      {/* Error state */}
      {error && (
        <div className="alert alert-error" role="alert">
          <span>⚠</span>
          <div>
            <strong>{error.status === 429 ? 'Rate limit reached' : 'Error'}</strong>
            <br />
            {error.message}
            {error.retryAfter && (
              <span style={{ fontSize: '0.8rem', display: 'block', marginTop: '0.25rem' }}>
                Try again in {error.retryAfter}s
              </span>
            )}
          </div>
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div className="card answer-card">
          <div className="card-header">
            <div className="card-title"><span>🤔</span> Thinking…</div>
          </div>
          <div className="card-body">
            <SkeletonAnswer />
          </div>
        </div>
      )}

      {/* Answer */}
      {result && !loading && (
        <div className="card answer-card">
          <div className="card-header">
            <div className="card-title"><span>💡</span> Answer</div>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              {result.response_mode === 'demo' && (
                <span className="demo-badge">⚡ Demo mode</span>
              )}
              {result.response_mode === 'gemini' && (
                <span className="badge badge-blue">Gemini</span>
              )}
              {result.response_mode === 'openai' && (
                <span className="badge badge-blue">OpenAI</span>
              )}
              {result.response_mode === 'anthropic' && (
                <span className="badge badge-blue">Anthropic</span>
              )}
            </div>
          </div>
          <div className="card-body">
            <p className="answer-text">{result.answer}</p>

            {sources && sources.length > 0 && (
              <div className="sources-section">
                <div className="sources-label">
                  Sources ({sources.length})
                </div>
                <div className="source-list">
                  {sources.map((src) => (
                    <SourceCard key={src.chunk_id} source={src} />
                  ))}
                </div>
              </div>
            )}

            {sources && sources.length === 0 && (
              <div className="alert alert-info" style={{ marginTop: '1rem' }}>
                <span>ℹ</span>
                <span>No relevant document chunks were found above the relevance threshold.</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
