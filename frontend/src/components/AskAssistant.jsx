import React, { useState, useRef, useCallback } from 'react';
import { askQuestion } from '../services/api';
import AnswerPane from './AnswerPane';
import { dedupeSources } from '../utils/dedupeSources';

export default function AskAssistant() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const textareaRef = useRef(null);

  const handleAsk = useCallback(async () => {
    const q = question.trim();
    if (!q || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await askQuestion(q);
      setResult(data);
    } catch (err) {
      setError({ message: err.message, status: err.status, retryAfter: err.retryAfter });
    } finally {
      setLoading(false);
    }
  }, [question, loading]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  const uniqueSourceCount = result ? dedupeSources(result.sources).length : 0;

  return (
    <div className="ask-panel">
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <span className="card-icon">💬</span>
            Ask the Assistant
          </div>
          {result && (
            <span className="badge badge-blue">{uniqueSourceCount} source{uniqueSourceCount !== 1 ? 's' : ''}</span>
          )}
        </div>
        <div className="card-body">
          <div className="question-input-wrap">
            <textarea
              id="question-input"
              ref={textareaRef}
              placeholder="Ask a question about your documents… (Enter to submit)"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              maxLength={2000}
              rows={2}
            />
            <button
              id="ask-btn"
              className="btn btn-primary"
              onClick={handleAsk}
              disabled={loading || !question.trim()}
              style={{ alignSelf: 'flex-end', minWidth: 90 }}
            >
              {loading ? <><span className="spinner" />Wait…</> : '✦ Ask'}
            </button>
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
            Enter to submit · Shift+Enter for newline · Max 2,000 characters
          </div>
        </div>
      </div>

      <AnswerPane result={result} loading={loading} error={error} />
    </div>
  );
}
