import React, { useState, useEffect, useCallback } from 'react';
import DocumentUpload from './components/DocumentUpload';
import DocumentList from './components/DocumentList';
import AskAssistant from './components/AskAssistant';
import { listDocuments, getHealth } from './services/api';
import './index.css';

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [docsTotal, setDocsTotal] = useState(0);
  const [docsLoading, setDocsLoading] = useState(true);
  const [docsError, setDocsError] = useState(null);
  const [healthy, setHealthy] = useState(null);

  const fetchDocs = useCallback(async () => {
    setDocsLoading(true);
    setDocsError(null);
    try {
      const data = await listDocuments(50, 0);
      setDocuments(data.documents);
      setDocsTotal(data.total);
    } catch (err) {
      setDocsError('Could not load documents. Is the backend running?');
    } finally {
      setDocsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocs();
    // Check health
    getHealth()
      .then(() => setHealthy(true))
      .catch(() => setHealthy(false));
  }, [fetchDocs]);

  const handleDocumentUploaded = (doc) => {
    fetchDocs();
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="header-brand">
            <div className="header-logo">🧠</div>
            <span className="header-title">Mini Knowledge Assistant</span>
          </div>
          <div className="header-status">
            <div className={`status-dot ${healthy === false ? 'offline' : ''}`} />
            {healthy === null ? 'Connecting…' : healthy ? 'Backend connected' : 'Backend offline'}
          </div>
        </div>
      </header>

      {/* Main layout */}
      <main>
        <div className="main-layout">
          {/* Left panel: document management */}
          <div className="doc-panel">
            <DocumentUpload onSuccess={handleDocumentUploaded} />
            <DocumentList
              documents={documents}
              loading={docsLoading}
              error={docsError}
              total={docsTotal}
            />
          </div>

          {/* Right panel: ask the assistant */}
          <AskAssistant />
        </div>
      </main>
    </div>
  );
}
