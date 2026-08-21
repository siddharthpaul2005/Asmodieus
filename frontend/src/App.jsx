import React, { useState, useEffect, useRef } from 'react';
import { Mic, Play, ChevronDown, Shield, AlertTriangle, BarChart3 } from 'lucide-react';
import './index.css';

// ─── Guardrail Engine ──────────────────────────────────────────────────────────
// Client-side guardrail for instant UX feedback.
// Backend src/guardrails/ is the source of truth for real enforcement.


// ── Guardrails Data ────────────────────────────────────────────────────────
// Loaded dynamically from backend /api/guardrails on mount

function runGuardrail(text, tiers) {
  const start = performance.now();

  for (const [tierKey, tier] of Object.entries(tiers)) {
    const flaggedWords = new Set();

    for (const regex of tier.patterns) {
      // Create a fresh copy so lastIndex never carries over between calls
      // (critical for /gi patterns — a shared regex object retains lastIndex
      // from the previous call and silently misses matches on re-entry).
      const r = new RegExp(regex.source, regex.flags);
      const matches = [...text.matchAll(r)];
      matches.forEach(m => flaggedWords.add(m[0].toLowerCase()));
    }

    if (flaggedWords.size > 0) {
      const latencyMs = parseFloat((performance.now() - start).toFixed(2));
      return {
        blocked: true,
        flaggedWords: [...flaggedWords],
        tier: tierKey,
        tierLabel: tier.label,
        tierColor: tier.color,
        category: tier.category,
        latencyMs,
      };
    }
  }

  return {
    blocked: false,
    flaggedWords: [],
    tier: null,
    tierLabel: null,
    tierColor: null,
    category: 'Clean',
    latencyMs: parseFloat((performance.now() - start).toFixed(2)),
  };
}

// ─── Blurred Transcript Renderer ──────────────────────────────────────────────
function BlurredTranscript({ text, flaggedWords }) {
  if (!flaggedWords || flaggedWords.length === 0) return <p>{text}</p>;

  const escaped = flaggedWords.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const pattern = new RegExp(`(${escaped.join('|')})`, 'gi');
  const parts = text.split(pattern);

  return (
    <p>
      {parts.map((part, i) => {
        const isFlag = flaggedWords.some(w => w.toLowerCase() === part.toLowerCase());
        return isFlag
          ? <span key={i} className="blurred-word" title="[flagged — hover to reveal]">{part}</span>
          : <span key={i}>{part}</span>;
      })}
    </p>
  );
}

// ─── Latency Simulator ────────────────────────────────────────────────────────
// NOTE: Values are simulated. Replace with backend-measured values when available.
function simulateLatency(blocked, guardrailMs) {
  if (blocked) {
    return { guardrail: guardrailMs, embedding: 0, dense: 0, bm25: 0, fusion: 0, rerank: 0, total: guardrailMs, isSimulated: true };
  }
  const g = parseFloat((Math.random() * 0.4 + 0.1).toFixed(1));
  const e = parseFloat((Math.random() * 14 + 8).toFixed(1));
  const d = parseFloat((Math.random() * 28 + 25).toFixed(1));
  const b = parseFloat((Math.random() * 22 + 10).toFixed(1));
  const f = parseFloat((Math.random() * 0.7 + 0.1).toFixed(1));
  const r = parseFloat((Math.random() * 2.5 + 0.5).toFixed(1));
  return { guardrail: g, embedding: e, dense: d, bm25: b, fusion: f, rerank: r, total: parseFloat((g + e + d + b + f + r).toFixed(1)), isSimulated: true };
}

// ─── App ──────────────────────────────────────────────────────────────────────
const App = () => {
  const [uiState, setUiState] = useState('ready');
  const [transcript, setTranscript] = useState('');
  const [textInput, setTextInput] = useState('');
  const [answer, setAnswer] = useState('');
  const [activeNav, setActiveNav] = useState('input');
  const [showInfo, setShowInfo] = useState(false);
  const [isTranscriptExpanded, setIsTranscriptExpanded] = useState(false);
  const [guardrailTiers, setGuardrailTiers] = useState({});

  // Guardrail + analytics state
  const [guardrailResult, setGuardrailResult] = useState(null);
  const [guardrailLog, setGuardrailLog] = useState([]);
  const [latencyLog, setLatencyLog] = useState([]);
  const [lastLatency, setLastLatency] = useState(null);

  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const audioStreamRef = useRef(null);
  const processorRef = useRef(null);
  const pipelineTimeoutRef = useRef(null);
  const typeWriterTimeoutRef = useRef(null);
  const analyticsRef = useRef(null);

  // ── Fetch dynamic guardrails ─────────────────────────────────────────────────
  useEffect(() => {
    fetch('http://localhost:8000/api/guardrails')
      .then(res => res.json())
      .then(data => {
        const parsedTiers = {};
        for (const [tier, obj] of Object.entries(data)) {
          let patterns = [];
          if (obj.regex_patterns) {
            patterns.push(...obj.regex_patterns.map(p => new RegExp(p, 'gi')));
          }
          if (obj.exact_words && obj.exact_words.length > 0) {
            const chunk = 500;
            for (let i = 0; i < obj.exact_words.length; i += chunk) {
              const slice = obj.exact_words.slice(i, i + chunk).map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
              patterns.push(new RegExp('(?:^|\\\\s|[.,!?;:])(' + slice.join('|') + ')(?=$|\\\\s|[.,!?;:])', 'gi'));
            }
          }
          parsedTiers[tier] = { ...obj, patterns };
        }
        setGuardrailTiers(parsedTiers);
        console.log('Loaded dynamic guardrails from backend.');
      })
      .catch(err => console.error('Failed to load guardrails:', err));
  }, []);

  // ── Audio / WebSocket ──────────────────────────────────────────────────────
  const stopRecording = () => {
    if (processorRef.current) { processorRef.current.disconnect(); processorRef.current = null; }
    if (audioStreamRef.current) { audioStreamRef.current.getTracks().forEach(t => t.stop()); audioStreamRef.current = null; }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') { audioContextRef.current.close(); audioContextRef.current = null; }
    if (wsRef.current) {
      // Prevent intentional closure from triggering error UI
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
      wsRef.current = null;
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioStreamRef.current = stream;
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioContext;
      await audioContext.audioWorklet.addModule('/audio-worklet.js');
      const source = audioContext.createMediaStreamSource(stream);
      const processor = new AudioWorkletNode(audioContext, 'pcm-downsampler');
      processorRef.current = processor;
      const ws = new WebSocket('ws://localhost:8000/ws/stt');
      wsRef.current = ws;

      ws.onopen = () => {
        setUiState('listening'); setActiveNav('input');
        processor.port.onmessage = (e) => { if (ws.readyState === WebSocket.OPEN) ws.send(e.data); };
        source.connect(processor);
        processor.connect(audioContext.destination);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        // ── Server sent an STT-unavailable error (API key missing, Sarvam down) ──
        if (data.stt_unavailable) {
          console.warn('STT unavailable:', data.error);
          stopRecording();
          setUiState('ready');
          setTranscript(`[Error: STT Backend Failed - ${data.error}]`);
          return;
        }

        if (data.text) {
          setTranscript(data.text);
          if (data.is_partial) {
            setUiState('transcribing');
          } else {
            stopRecording();
            // ── If backend already ran guardrail, use its authoritative result ──
            if (data.guardrail?.blocked) {
              const g = data.guardrail;
              const backendResult = {
                blocked: true,
                flaggedWords: g.flagged_words ?? [],
                tier: g.tier,
                tierLabel: g.tier_label,
                tierColor: g.tier === 'TIER_1' ? '#ff2a2a' : g.tier === 'TIER_2' ? '#ff8c00' : '#ffb020',
                category: g.category,
                latencyMs: g.latency_ms ?? 0,
              };
              setGuardrailResult(backendResult);
              const logEntry = {
                id: Date.now(), timestamp: new Date().toLocaleTimeString(),
                query: data.text, flaggedWords: backendResult.flaggedWords,
                tier: backendResult.tierLabel, tierColor: backendResult.tierColor,
                category: backendResult.category, passed: false,
              };
              setGuardrailLog(prev => [logEntry, ...prev]);
              const lat = { guardrail: g.latency_ms ?? 0, embedding: 0, dense: 0, bm25: 0, fusion: 0, rerank: 0, total: g.latency_ms ?? 0, isSimulated: false };
              setLastLatency(lat);
              setLatencyLog(prev => [lat, ...prev]);
              setUiState('blocked');
              setAnswer('__BLOCKED__');
              setActiveNav('guardrail');
            } else {
              // Clean final transcript — client-side guardrail will run via useEffect
              setUiState('ready');
            }
          }
        }
      };

      // ── Server closed the connection (Sarvam error, timeout, API key missing) ──
      // ws.onclose fires when the server closes the socket from its side.
      // Without this handler the client freezes in "listening" state forever.
      ws.onclose = (evt) => {
        // Normal close codes (1000 = normal, 1001 = going away) after a
        // successful session don't need simulation.
        if (evt.code === 1000 || evt.code === 1001) return;
        console.warn('WebSocket closed unexpectedly (code', evt.code, ')');
        stopRecording();
        setUiState('ready');
        setTranscript(`[Error: Backend connection closed unexpectedly (code ${evt.code})]`);
      };

      ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        stopRecording();
        setUiState('ready');
        setTranscript('[Error: Failed to connect to WebSocket backend]');
      };
    } catch (e) {
      console.error('Mic error:', e);
      setUiState('ready');
      setTranscript('[Error: Could not access microphone. Check permissions.]');
    }
  };

  useEffect(() => () => { clearTimeout(pipelineTimeoutRef.current); clearTimeout(typeWriterTimeoutRef.current); stopRecording(); }, []);

  // Trigger pipeline when transcription finalises
  useEffect(() => {
    if (uiState === 'ready' && transcript.trim().length > 0 && !answer && !guardrailResult) {
      processQuery(transcript);
    }
  }, [uiState, transcript, answer, guardrailResult]);

  // ── Query processing ───────────────────────────────────────────────────────
  const processQuery = async (query) => {
    // 1. Guardrail Check (Client-side)
    const result = runGuardrail(query, guardrailTiers);
    setGuardrailResult(result);

    // Briefly show guardrail stage activating
    setUiState('transcribing');
    setActiveNav('guardrail');

    if (result.blocked) {
      const logEntry = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        query,
        flaggedWords: result.flaggedWords,
        tier: result.tierLabel,
        tierColor: result.tierColor,
        category: result.category,
        passed: false,
      };
      setGuardrailLog(prev => [logEntry, ...prev]);
      const lat = simulateLatency(true, result.latencyMs);
      setLastLatency(lat);
      setLatencyLog(prev => [lat, ...prev]);

      pipelineTimeoutRef.current = setTimeout(() => {
        setUiState('blocked');
        setAnswer('__BLOCKED__');
      }, 700);
      return;
    }

    // Clean query — proceed through RAG
    pipelineTimeoutRef.current = setTimeout(() => {
      setUiState('retrieving');
      setActiveNav('retrieval');

      pipelineTimeoutRef.current = setTimeout(() => {
        const lat = simulateLatency(false, result.latencyMs);
        setLastLatency(lat);
        setLatencyLog(prev => [lat, ...prev]);

        setGuardrailLog(prev => [{
          id: Date.now(),
          timestamp: new Date().toLocaleTimeString(),
          query,
          flaggedWords: [],
          tier: null, tierColor: null,
          category: 'Clean',
          passed: true,
        }, ...prev]);

        setUiState('answering');
        setActiveNav('answer');
        typeWriter(
          'Based on the retrieved context, this signifies a potential risk pattern originating from the subnet. We recommend isolating the affected sector and initiating automated containment protocols immediately.',
          setAnswer, 25
        );
      }, 1800);
    }, 1200);
  };

  const resetUI = () => {
    clearTimeout(pipelineTimeoutRef.current);
    clearTimeout(typeWriterTimeoutRef.current);
    setTranscript(''); setAnswer('');
    setUiState('ready'); setActiveNav('input');
    setIsTranscriptExpanded(false);
    setGuardrailResult(null);
  };

  const typeWriter = (text, setFunc, speed = 30) => {
    setFunc(''); let i = 0;
    const type = () => { if (i < text.length) { setFunc(p => p + text.charAt(i)); i++; typeWriterTimeoutRef.current = setTimeout(type, speed); } };
    type();
  };

  const simulateFullInteraction = () => {
    resetUI();
    setUiState('listening'); setActiveNav('input');
    pipelineTimeoutRef.current = setTimeout(() => {
      const mockQuery = 'Analyze recent network traffic for anomalous propagation.';
      setUiState('transcribing'); setActiveNav('stt');
      typeWriter(mockQuery, setTranscript, 40);
      pipelineTimeoutRef.current = setTimeout(() => { setTranscript(mockQuery); setUiState('ready'); }, mockQuery.length * 40 + 500);
    }, 1500);
  };

  const handleMicClick = () => {
    if (uiState === 'listening') { stopRecording(); resetUI(); }
    else { resetUI(); startRecording(); }
  };

  const handleTextSubmit = (e) => {
    e.preventDefault();
    if (!textInput.trim() || uiState === 'listening') return;

    resetUI();
    setUiState('transcribing');
    setActiveNav('stt');

    // Simulate STT typing effect for consistency
    typeWriter(textInput, setTranscript, 20);
    pipelineTimeoutRef.current = setTimeout(() => {
      setTranscript(textInput);
      setUiState('ready');
      setTextInput('');
    }, textInput.length * 20 + 300);
  };

  const getStatusText = () => {
    switch (uiState) {
      case 'ready': return 'Tap mic to speak or simulate...';
      case 'listening': return 'Listening to your query...';
      case 'retrieving': return 'Chunking & Retrieving Context...';
      default: return '';
    }
  };

  // ── Analytics computations ─────────────────────────────────────────────────
  const blockedCount = guardrailLog.filter(e => !e.passed).length;
  const blockedPct = guardrailLog.length > 0 ? Math.round((blockedCount / guardrailLog.length) * 100) : 0;
  const cleanLatencies = latencyLog.filter((_, i) => guardrailLog[i]?.passed).map(l => l.total);
  const avgRagMs = cleanLatencies.length > 0 ? (cleanLatencies.reduce((a, b) => a + b, 0) / cleanLatencies.length).toFixed(1) : '—';

  const isBlocked = uiState === 'blocked';
  const currentFlags = guardrailResult?.flaggedWords ?? [];

  const LATENCY_STAGES = ['guardrail', 'embedding', 'dense', 'bm25', 'fusion', 'rerank'];

  // ─────────────────────────────────────────────────────────────────────────────
  return (
    <>
      <div className="background-effects">
        <div className="glow-orb" />
      </div>

      {/* ── Hero page ── */}
      <main className="container">
        <header className="top-nav">
          <div className="super-title" style={{ margin: 0 }}>ASMODIEUS</div>
          <div className="nav-pill">
            <span className={activeNav === 'input' ? 'active' : ''}>Voice Input</span>
            <span className={activeNav === 'stt' ? 'active' : ''}>STT</span>
            <span className={`${activeNav === 'guardrail' ? 'active' : ''} ${isBlocked ? 'danger-stage' : ''}`}>Guardrail</span>
            <span className={activeNav === 'retrieval' ? 'active' : ''}>Retrieval</span>
            <span className={activeNav === 'answer' ? 'active' : ''}>Generation</span>
          </div>
        </header>

        <section className="hero-section">
          <div className="subtitle">AI-NATIVE RAG PLATFORM</div>
          <h1 className="main-title" style={{ opacity: uiState === 'listening' ? 0.15 : 1, transform: uiState === 'listening' ? 'scale(0.95)' : 'scale(1)' }}>
            Speak Query.<br />Get Context.
          </h1>

          <div className="interactive-center">
            <div className={`ai-entity ${uiState === 'listening' ? 'listening' : ''} ${['transcribing', 'retrieving'].includes(uiState) ? 'processing' : ''} ${isBlocked ? 'entity-blocked' : ''}`}>
              <div className="rings">
                <div className="ring ring-1" /><div className="ring ring-2" /><div className="ring ring-3" />
              </div>
              <button className={`mic-btn ${isBlocked ? 'mic-blocked' : ''}`} onClick={handleMicClick}>
                {isBlocked ? <Shield size={32} /> : <Mic size={32} />}
              </button>
            </div>
          </div>

          <form onSubmit={handleTextSubmit} className="text-input-form" style={{ width: '100%', maxWidth: '400px', display: 'flex', justifyContent: 'center', marginBottom: '2rem' }}>
            <input
              type="text"
              className="glass-input"
              placeholder="...or type your query here"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              disabled={uiState === 'listening' || uiState === 'transcribing' || uiState === 'retrieving'}
              style={{
                width: '100%',
                padding: '12px 20px',
                borderRadius: '30px',
                border: '1px solid var(--glass-border)',
                background: 'rgba(20, 30, 25, 0.4)',
                color: 'white',
                backdropFilter: 'blur(10px)',
                outline: 'none',
                fontSize: '1rem',
                fontFamily: 'inherit',
                textAlign: 'center',
                transition: 'all 0.3s ease',
                opacity: (uiState === 'ready' || uiState === 'answering' || uiState === 'blocked') ? 1 : 0.4
              }}
            />
          </form>

          <div className="content-display">
            <div className="status-text" style={{ opacity: ['ready', 'listening', 'retrieving'].includes(uiState) ? 1 : 0 }}>
              {getStatusText()}
            </div>

            {/* Transcript box */}
            <div
              className={[
                'transcript-box',
                ['ready', 'listening'].includes(uiState) ? 'hidden' : '',
                ((uiState === 'answering' || isBlocked) && !isTranscriptExpanded) ? 'compact' : '',
                (uiState === 'answering' || isBlocked) ? 'clickable' : '',
                isBlocked ? 'flagged' : '',
              ].join(' ')}
              style={{
                borderColor: uiState === 'retrieving' ? 'var(--accent)' : isBlocked ? 'rgba(255,42,42,0.5)' : 'var(--glass-border)',
                boxShadow: uiState === 'retrieving' ? '0 0 30px rgba(255,69,0,0.2)' : isBlocked ? '0 0 24px rgba(255,42,42,0.18)' : '0 10px 40px rgba(0,0,0,0.3)',
              }}
              onClick={() => { if (uiState === 'answering' || isBlocked) setIsTranscriptExpanded(p => !p); }}
            >
              <div className="label-row">
                <span className="label">Transcribed Query</span>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  {isBlocked && <span className="badge danger-badge">⚠ Flagged</span>}
                  <span className="badge">Speech-to-Text</span>
                </div>
              </div>
              <BlurredTranscript text={transcript} flaggedWords={currentFlags} />
              {(uiState === 'answering' || isBlocked) && (
                <div className="expand-hint">{isTranscriptExpanded ? 'Tap to collapse' : 'Tap to expand'}</div>
              )}
            </div>

            {/* Clean answer box */}
            <div className={`answer-box ${uiState !== 'answering' ? 'hidden' : ''}`}>
              <div className="label-row">
                <span className="label">Synthesized Response</span>
                <span className="badge accent">Vector DB + LLM</span>
              </div>
              <p>{answer}</p>
            </div>

            {/* Refusal box */}
            <div className={`refusal-box ${!isBlocked ? 'hidden' : ''}`}>
              <div className="label-row">
                <span className="label">Guardrail Response</span>
                {guardrailResult && (
                  <span className="badge danger-badge" style={{ borderColor: guardrailResult.tierColor, color: guardrailResult.tierColor }}>
                    {guardrailResult.tierLabel} · {guardrailResult.category}
                  </span>
                )}
              </div>
              <div className="refusal-content">
                <Shield size={28} className="refusal-icon" />
                <p>It is not ethically correct to answer this query. This request has been flagged and logged by the Asmodieus guardrail system.</p>
              </div>
              <button className="reset-btn" onClick={resetUI}>Ask another question →</button>
            </div>
          </div>
        </section>

        {/* Scroll hint — appears once analytics have data */}
        {guardrailLog.length > 0 && (
          <div className="scroll-hint" onClick={() => analyticsRef.current?.scrollIntoView({ behavior: 'smooth' })}>
            <span>Session Analytics</span>
            <ChevronDown size={16} />
          </div>
        )}

        <footer className="bottom-bar">
          <div>
            <p className="description">Identify weak signals, map attack paths,<br />and stop incidents before they escalate<br /> via Voice.</p>
            <p className="credits">exclusively built for HHGoa 26</p>
          </div>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            {guardrailLog.length > 0 && (
              <button className="secondary-btn" style={{ display: 'flex', alignItems: 'center', gap: 8 }}
                onClick={() => analyticsRef.current?.scrollIntoView({ behavior: 'smooth' })}>
                <BarChart3 size={15} /> Analytics
              </button>
            )}
            <button className="primary-btn red" onClick={() => setShowInfo(true)}>
              <Play size={18} fill="currentColor" /> Info Dump
            </button>
          </div>
        </footer>
      </main>

      {/* ── Analytics Section ── */}
      {guardrailLog.length > 0 && (
        <section className="analytics-section" ref={analyticsRef}>
          <div className="analytics-header">
            <BarChart3 size={20} />
            <h2>Session Analytics</h2>
            <span className="analytics-subtitle">· {guardrailLog.length} quer{guardrailLog.length === 1 ? 'y' : 'ies'} this session</span>
          </div>

          {/* Summary counters */}
          <div className="stat-counter-row">
            <div className="stat-tile">
              <span className="stat-value">{guardrailLog.length}</span>
              <span className="stat-label">Total Queries</span>
            </div>
            <div className="stat-tile stat-danger">
              <span className="stat-value">{blockedPct}<small>%</small></span>
              <span className="stat-label">Blocked</span>
            </div>
            <div className="stat-tile stat-accent">
              <span className="stat-value">{avgRagMs}{avgRagMs !== '—' ? <small> ms</small> : ''}</span>
              <span className="stat-label">Avg RAG Latency</span>
            </div>
            <div className="stat-tile">
              <span className="stat-value">{guardrailLog.length - blockedCount}</span>
              <span className="stat-label">Passed</span>
            </div>
          </div>

          <div className="analytics-grid">
            {/* Latency breakdown card — matches reference image */}
            {lastLatency && (
              <div className="latency-card">
                <p className="latency-card-label">LATENCY</p>
                <div className="latency-card-hero">
                  <span className="latency-big-num">{lastLatency.total}</span>
                  <span className="latency-unit">MS RAG</span>
                </div>
                <p className="latency-subtitle">
                  {isBlocked ? 'Blocked at guardrail · RAG skipped' : 'Warm retrieval path · STT separate'}
                  &nbsp;·&nbsp;<span className="sim-label">simulated</span>
                </p>
                <div className="latency-divider" />
                <div className="latency-table">
                  <div className="latency-header-row">
                    <span>STAGE</span>
                    <span>MS</span>
                  </div>
                  {LATENCY_STAGES.map(s => (
                    <div className="latency-row" key={s}>
                      <span>{s}</span>
                      <span>{lastLatency[s] === 0 ? '0.0' : lastLatency[s].toFixed(1)}</span>
                    </div>
                  ))}
                  <div className="latency-row latency-total">
                    <span>total_rag</span>
                    <span>{lastLatency.total.toFixed(1)}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Guardrail events log */}
            <div className="guardrail-log-card">
              <div className="log-card-header">
                <AlertTriangle size={15} />
                <span>Guardrail Events Log</span>
                <span className="log-count">{blockedCount} flagged</span>
              </div>
              <div className="log-entries">
                {guardrailLog.map(entry => (
                  <div key={entry.id} className={`log-entry ${entry.passed ? 'log-clean' : 'log-blocked'}`}>
                    <div className="log-entry-top">
                      <span className="log-time">{entry.timestamp}</span>
                      {entry.tier
                        ? <span className="tier-badge" style={{ borderColor: entry.tierColor, color: entry.tierColor }}>{entry.tier}</span>
                        : <span className="tier-badge tier-pass">PASS</span>
                      }
                    </div>
                    <div className="log-query">
                      <BlurredTranscript text={entry.query} flaggedWords={entry.flaggedWords} />
                    </div>
                    {entry.flaggedWords.length > 0 && (
                      <div className="log-flags">
                        <span className="log-cat">{entry.category}</span>
                        {entry.flaggedWords.slice(0, 3).map((w, i) => (
                          <span key={i} className="flag-chip">{w.slice(0, 2)}***</span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Info Modal */}
      {showInfo && (
        <div className="modal-overlay" onClick={() => setShowInfo(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="close-btn" onClick={() => setShowInfo(false)}>X</button>
            <h3>RAG & Voice Architecture</h3>
            <p><strong>Retrieval-Augmented Generation (RAG)</strong> retrieves facts from an external vector database to ground LLMs on accurate context before generating a response.</p>
            <p>By combining a <strong>Voice Transcribing Model</strong> (Sarvam Saaras v3 STT) with a RAG pipeline, Asmodieus enables frictionless, hands-free interaction.</p>
            <p>The <strong>Guardrail System</strong> fires before every query — Tier 1 blocks harmful/profane content, Tier 2 blocks safety violations and prompt injection, Tier 3 blocks off-topic queries. Blocked queries never reach the RAG pipeline.</p>
          </div>
        </div>
      )}
    </>
  );
};

export default App;
