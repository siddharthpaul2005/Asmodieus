import React, { useState, useEffect, useRef } from 'react';
import { ArrowLeft, Mic, Play } from 'lucide-react';
import './index.css';

const App = () => {
  const [uiState, setUiState] = useState('ready'); // ready, listening, transcribing, retrieving, answering
  const [transcript, setTranscript] = useState('');
  const [answer, setAnswer] = useState('');
  const [activeNav, setActiveNav] = useState('input');
  const [showInfo, setShowInfo] = useState(false);

  const recognitionRef = useRef(null);
  const pipelineTimeoutRef = useRef(null);
  const typeWriterTimeoutRef = useRef(null);

  useEffect(() => {
    // Check for Speech API support (Web Speech API)
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        setUiState('listening');
        setActiveNav('input');
      };

      recognition.onresult = (event) => {
        const current = event.resultIndex;
        const currentTranscript = event.results[current][0].transcript;

        setTranscript(currentTranscript);
        setUiState((prev) => (prev === 'listening' ? 'transcribing' : prev));
      };

      recognition.onspeechend = () => {
        recognition.stop();
      };

      recognition.onend = () => {
        setUiState((currentUiState) => {
          if (currentUiState === 'listening' || currentUiState === 'transcribing') {
            // Need to get latest transcript from ref or state if possible, but React closure might have old state
            // It's handled by calling runPipelineSimulation in a useEffect watching for the end, or here.
            // We'll rely on the transcript state.
            return 'ready';
          }
          return currentUiState;
        });
      };

      recognition.onerror = (event) => {
        console.error('Speech recognition error', event.error);
        simulateFullInteraction();
      };

      recognitionRef.current = recognition;
    }

    return () => {
      clearTimeout(pipelineTimeoutRef.current);
      clearTimeout(typeWriterTimeoutRef.current);
    };
  }, []);

  // Effect to trigger pipeline when transcription is done
  useEffect(() => {
    if (uiState === 'ready' && transcript.trim().length > 0 && !answer) {
      runPipelineSimulation(transcript);
    }
  }, [uiState, transcript, answer]);

  const resetUI = () => {
    clearTimeout(pipelineTimeoutRef.current);
    clearTimeout(typeWriterTimeoutRef.current);
    setTranscript('');
    setAnswer('');
    setUiState('ready');
    setActiveNav('input');
  };

  const typeWriter = (text, setFunc, speed = 30) => {
    setFunc('');
    let i = 0;
    const type = () => {
      if (i < text.length) {
        setFunc((prev) => prev + text.charAt(i));
        i++;
        typeWriterTimeoutRef.current = setTimeout(type, speed);
      }
    };
    type();
  };

  const runPipelineSimulation = (query) => {
    setUiState('transcribing');
    setActiveNav('stt');

    // Simulate Vector DB Retrieval delay
    pipelineTimeoutRef.current = setTimeout(() => {
      setUiState('retrieving');
      setActiveNav('retrieval');

      // Simulate Generation delay
      pipelineTimeoutRef.current = setTimeout(() => {
        setUiState('answering');
        setActiveNav('answer');

        const mockAnswer = "Based on the retrieved context, this signifies a potential risk pattern originating from the subnet. We recommend isolating the affected sector and initiating automated containment protocols immediately.";
        typeWriter(mockAnswer, setAnswer, 25);

      }, 1800);
    }, 1200);
  };

  const simulateFullInteraction = () => {
    resetUI();
    setUiState('listening');
    setActiveNav('input');

    pipelineTimeoutRef.current = setTimeout(() => {
      setUiState('transcribing');
      setActiveNav('stt');
      const mockQuery = "Analyze recent network traffic for anomalous propagation.";

      typeWriter(mockQuery, setTranscript, 40);

      pipelineTimeoutRef.current = setTimeout(() => {
        runPipelineSimulation(mockQuery);
      }, mockQuery.length * 40 + 500);

    }, 1500);
  };

  const handleMicClick = () => {
    if (uiState === 'listening') {
      if (recognitionRef.current) recognitionRef.current.stop();
      resetUI();
    } else {
      resetUI();
      if (recognitionRef.current) {
        try {
          recognitionRef.current.start();
        } catch (e) {
          simulateFullInteraction();
        }
      } else {
        simulateFullInteraction();
      }
    }
  };

  const getStatusText = () => {
    switch (uiState) {
      case 'ready': return 'Tap mic to speak or simulate...';
      case 'listening': return 'Listening to your query...';
      case 'retrieving': return 'Chunking & Retrieving Context...';
      default: return '';
    }
  };

  return (
    <>
      <div className="background-effects">
        <div className="glow-orb"></div>
      </div>

      <main className="container">
        {/* Top Navigation Area */}
        <header className="top-nav">
          <div className="super-title" style={{ margin: 0 }}>ASMODIEUS</div>

          <div className="nav-pill">
            <span className={activeNav === 'input' ? 'active' : ''}>Voice Input</span>
            <span className={activeNav === 'stt' ? 'active' : ''}>STT</span>
            <span className={activeNav === 'retrieval' ? 'active' : ''}>Retrieval</span>
            <span className={activeNav === 'answer' ? 'active' : ''}>Generation</span>
          </div>
        </header>

        {/* Main Content Area */}
        <section className="hero-section">
          <div className="subtitle">AI-NATIVE RAG PLATFORM</div>
          <h1
            className="main-title"
            style={{
              opacity: uiState === 'listening' ? 0.15 : 1,
              transform: uiState === 'listening' ? 'scale(0.95)' : 'scale(1)'
            }}
          >
            Speak Query.<br />Get Context.
          </h1>

          <div className="interactive-center">
            <div className={`ai-entity ${uiState === 'listening' ? 'listening' : ''} ${['transcribing', 'retrieving'].includes(uiState) ? 'processing' : ''}`}>
              <div className="rings">
                <div className="ring ring-1"></div>
                <div className="ring ring-2"></div>
                <div className="ring ring-3"></div>
              </div>
              <button className="mic-btn" onClick={handleMicClick}>
                <Mic size={32} />
              </button>
            </div>
          </div>

          {/* Display for pipeline outputs */}
          <div className="content-display">
            <div className="status-text" style={{ opacity: ['ready', 'listening', 'retrieving'].includes(uiState) ? 1 : 0 }}>
              {getStatusText()}
            </div>

            <div
              className={`transcript-box ${['ready', 'listening'].includes(uiState) ? 'hidden' : ''}`}
              style={{
                borderColor: uiState === 'retrieving' ? 'var(--accent)' : (uiState === 'answering' ? 'var(--glass-border)' : 'var(--glass-border)'),
                boxShadow: uiState === 'retrieving' ? '0 0 30px rgba(255, 69, 0, 0.2)' : (uiState === 'answering' ? 'none' : '0 10px 40px rgba(0,0,0,0.3)'),
                opacity: uiState === 'answering' ? 0.3 : 1,
                transform: uiState === 'answering' ? 'translateY(-30px) scale(0.95)' : 'translateY(0)'
              }}
            >
              <div className="label-row">
                <span className="label">Transcribed Query</span>
                <span className="badge">Speech-to-Text</span>
              </div>
              <p>{transcript}</p>
            </div>

            <div className={`answer-box ${uiState !== 'answering' ? 'hidden' : ''}`}>
              <div className="label-row">
                <span className="label">Synthesized Response</span>
                <span className="badge accent">Vector DB + LLM</span>
              </div>
              <p>{answer}</p>
            </div>
          </div>
        </section>

        {/* Bottom Bar */}
        <footer className="bottom-bar">
          <div>
            <p className="description">Identify weak signals, map attack paths,<br />and stop incidents before they escalate via Voice.</p>
            <p className="credits">exclusively built for HHGoa 26</p>
          </div>
          <button className="primary-btn red" onClick={() => setShowInfo(true)}>
            <Play size={18} fill="currentColor" />
            Info Dump
          </button>
        </footer>
      </main>

      {/* Information Modal */}
      {showInfo && (
        <div className="modal-overlay" onClick={() => setShowInfo(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="close-btn" onClick={() => setShowInfo(false)}>X</button>
            <h3>RAG & Voice Architecture</h3>
            <p>
              <strong>Retrieval-Augmented Generation (RAG)</strong> is a cutting-edge AI architecture that retrieves facts from an external, localized vector database to ground Large Language Models (LLMs) on highly accurate, up-to-date context before generating a response.
            </p>
            <p>
              By combining a <strong>Voice Transcribing Model</strong> (Speech-to-Text) with a RAG pipeline, Asmodieus allows for frictionless, hands-free interaction. Your spoken query is instantly transcribed, semantic context is retrieved, and a grounded answer is synthesized in real-time.
            </p>
          </div>
        </div>
      )}
    </>
  );
};

export default App;
