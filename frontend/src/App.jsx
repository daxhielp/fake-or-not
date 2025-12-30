import { useState } from 'react'
import './App.css'

function App() {
  const [url, setUrl] = useState('')
  const [result, setResult] = useState(null)
  const [confidence, setConfidence] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async () => {
    if (!url.trim()) return
    
    setLoading(true)
    setError(null)
    setResult(null)
    setConfidence(null)

    console.log(url);

    try {
      const response = await fetch('https://fake-or-not-qc8v.vercel.app/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      })

      if (!response.ok) {
        throw new Error('Failed to get prediction')
      }

      const data = await response.json()
      setResult(data.prediction)
      setConfidence(data.confidence)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setUrl('')
    setResult(null)
    setConfidence(null)
    setError(null)
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>Fake News Detector</h1>
        <p>AI-powered verification tool</p>
      </header>

      <main className="main-content">
        <div className="input-section">
          <input
            type="url"
            className="news-input"
            placeholder="Paste the news article URL here to verify..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <div className="button-group">
            <button 
              className="analyze-btn" 
              onClick={handleSubmit}
              disabled={loading || !url.trim()}
            >
              {loading ? 'Analyzing...' : 'Verify News'}
            </button>
            <button className="clear-btn" onClick={handleClear}>
              Clear
            </button>
          </div>
        </div>

        {error && <div className="error-message">{error}</div>}

        {result && (
          <div className={`result-card ${result.toLowerCase()}`}>
            <h2>Prediction Result</h2>
            <div className="result-badge">
              {result}
            </div>
            {confidence !== null && (
              <div className="confidence-section">
                <p className="confidence-label">Fake Probability: <strong>{confidence}%</strong></p>
                <div className="progress-bar-bg">
                  <div 
                    className="progress-bar-fill" 
                    style={{ width: `${confidence}%`, backgroundColor: confidence > 50 ? 'var(--danger-color)' : 'var(--success-color)' }}
                  ></div>
                </div>
              </div>
            )}
            <p className="result-description">
              {result === 'REAL' 
                ? 'This article appears to be credible based on our analysis.'
                : 'Caution: This article shows signs of being fake news.'}
            </p>
          </div>
        )}
      </main>
      
      <footer className="footer">
        <p>Powered by Machine Learning &bull; React &bull; FastAPI</p>
      </footer>
    </div>
  )
}

export default App